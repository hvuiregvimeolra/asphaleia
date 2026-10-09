#!/usr/bin/env python3
"""
ASPHALEIA : Maintenance prédictive (filière IA & Data)
=======================================================
Pipeline complet :
  1. Simulation des capteurs (DHT22 -> température, MQ-2 -> gaz) avec incidents injectés
  2. Feature engineering sur séries temporelles (fenêtres glissantes, pentes, corrélation)
  3. Entraînement d'un Isolation Forest (non supervisé) sur données saines uniquement
  4. Calibration du seuil d'alerte sur une période de validation saine
  5. Évaluation (niveau échantillon + niveau incident + délai d'anticipation)
     et comparaison avec un seuil statique (if temp > 40 ...), interdit par le sujet
  6. Export : CSV, modèle .joblib, métriques .json, graphiques .png, rapport .md

Usage :
    pip install numpy pandas scikit-learn matplotlib joblib
    python asphaleia_predictive_maintenance.py
Sorties dans ./outputs_asphaleia/
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler


#--------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
OUT = Path(__file__).resolve().parent / "outputs_asphaleia"   # dossier à côté du script
OUT.mkdir(exist_ok=True)
 
SAMPLE_SEC = 10                      # 1 mesure toutes les 10 s (ESP8266)
PER_MIN = 60 // SAMPLE_SEC
N_DAYS = 14
N = N_DAYS * 24 * 3600 // SAMPLE_SEC
 
TRAIN_DAYS, VAL_DAYS = 6, 2          # jours 0-6 : train | 6-8 : validation | 8-14 : test
T_CRIT, GAS_CRIT = 40.0, 300.0       # seuils critiques physiques (°C, ppm)
FALSE_ALARM_QUANTILE = 0.001         # 0,1 % des échantillons sains au-dessus du seuil
PERSISTENCE = 6                      # alerte si 6 détections consécutives (~1 min)
 
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# 1. Simulation des données
# --------------------------------------------------------------------------- #
def ar1(n, phi, sigma):
    """Bruit autorégressif AR(1) : les vrais capteurs ne sont pas du bruit blanc."""
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def simulate():
    idx = pd.date_range("2026-10-05 00:00:00", periods=N, freq=f"{SAMPLE_SEC}s")
    hours = np.asarray(idx.hour + idx.minute / 60, dtype=float)

    # Température : cycle jour/nuit + bruit corrélé + bruit de mesure DHT22
    temp = 25 + 3.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp += ar1(N, 0.995, 0.03) + rng.normal(0, 0.15, N)

    # Gaz MQ-2 (ppm) : ligne de base, léger couplage thermique, bruit
    gas = 180 + 1.5 * (temp - 25) + ar1(N, 0.98, 0.8) + rng.normal(0, 3.0, N)

    label = np.zeros(N, dtype=int)       # 0 sain | 1 précurseur | 2 critique
    event_id = np.full(N, -1, dtype=int)
    event_type = np.array([""] * N, dtype=object)
    recovery = np.zeros(N, dtype=bool)   # retour à la normale + fenêtres glissantes encore perturbées (exclu de l'évaluation)

    # Incidents injectés uniquement dans la période de test (jours 8 à 14)
    day = 24 * 3600 // SAMPLE_SEC
    events = [
        # (type, jour, heure_début)
        ("drift", 8.4, None), ("leak", 9.6, None), ("heat", 10.7, None),
        ("drift", 11.5, None), ("leak", 12.6, None), ("drift", 13.3, None),
    ]
    for k, (etype, d, _) in enumerate(events):
        s = int(d * day)
        if etype == "drift":
            # Précurseur lent (40 min) : +8 °C et +25 ppm, puis phase critique (10 min)
            n1, n2 = 40 * PER_MIN, 10 * PER_MIN
            ramp1 = np.linspace(0, 1, n1)
            ramp2 = np.linspace(0, 1, n2) ** 2
            temp[s:s + n1] += 8 * ramp1
            gas[s:s + n1] += 25 * ramp1
            temp[s + n1:s + n1 + n2] += 8 + 10 * ramp2
            gas[s + n1:s + n1 + n2] += 25 + 250 * ramp2
            temp[s + n1 + n2:s + n1 + n2 + 30 * PER_MIN] += 18 * np.linspace(1, 0, 30 * PER_MIN)
            gas[s + n1 + n2:s + n1 + n2 + 30 * PER_MIN] += 275 * np.linspace(1, 0, 30 * PER_MIN)
            label[s:s + n1] = 1
            label[s + n1:s + n1 + n2] = 2
            end = s + n1 + n2
            recovery[end:end + 150 * PER_MIN] = True
        elif etype == "leak":
            # Fuite de gaz : montée exponentielle sur 8 min, précurseur court
            n1, n2 = 3 * PER_MIN, 5 * PER_MIN
            t_ = np.linspace(0, 1, n1 + n2)
            gas[s:s + n1 + n2] += 420 * (np.exp(3.5 * t_) - 1) / (np.exp(3.5) - 1)
            temp[s:s + n1 + n2] += 1.5 * t_
            gas[s + n1 + n2:s + n1 + n2 + 20 * PER_MIN] += 420 * np.linspace(1, 0, 20 * PER_MIN)
            label[s:s + n1] = 1
            label[s + n1:s + n1 + n2] = 2
            end = s + n1 + n2
            recovery[end:end + 150 * PER_MIN] = True
        else:  # heat : surchauffe thermique pure, sans gaz
            n1, n2 = 15 * PER_MIN, 8 * PER_MIN
            temp[s:s + n1] += 7 * np.linspace(0, 1, n1)
            temp[s + n1:s + n1 + n2] += 7 + 12 * np.linspace(0, 1, n2)
            temp[s + n1 + n2:s + n1 + n2 + 30 * PER_MIN] += 19 * np.linspace(1, 0, 30 * PER_MIN)
            label[s:s + n1] = 1
            label[s + n1:s + n1 + n2] = 2
            end = s + n1 + n2
            recovery[end:end + 150 * PER_MIN] = True
        event_id[s:end] = k
        event_type[s:end] = etype

    df = pd.DataFrame({
        "timestamp": idx, "temperature": temp.round(2), "gas_ppm": gas.round(1),
        "label": label, "recovery": recovery, "event_id": event_id, "event_type": event_type,
    })
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #
FEATURES = [
    "temperature", "gas_ppm",
    "temp_mean_5m", "gas_mean_5m",
    "temp_std_5m", "gas_std_5m",
    "temp_slope_15m", "gas_slope_15m",
    "temp_dev_1h", "gas_dev_1h",
    "corr_temp_gas_15m",
]


def build_features(df):
    d = df.copy()
    w5, w15, w60 = 5 * PER_MIN, 15 * PER_MIN, 60 * PER_MIN
    for c, p in (("temperature", "temp"), ("gas_ppm", "gas")):
        d[f"{p}_mean_5m"] = d[c].rolling(w5, min_periods=w5).mean()
        d[f"{p}_std_5m"] = d[c].rolling(w5, min_periods=w5).std()
        # Pente en unité/minute sur 15 min (sur la moyenne lissée)
        d[f"{p}_slope_15m"] = (d[f"{p}_mean_5m"] - d[f"{p}_mean_5m"].shift(w15)) / 15
        # Écart à la moyenne de la dernière heure (décalée pour ne pas inclure l'instant courant)
        d[f"{p}_dev_1h"] = d[f"{p}_mean_5m"] - d[c].rolling(w60, min_periods=w60).mean().shift(w5)
    d["corr_temp_gas_15m"] = d["temperature"].rolling(w15, min_periods=w15).corr(d["gas_ppm"])
    d = d.dropna().reset_index(drop=True)
    return d


# --------------------------------------------------------------------------- #
# 3-4. Entraînement + calibration
# --------------------------------------------------------------------------- #
def persistence_filter(flags, k):
    """Garde une alerte uniquement si k détections consécutives (anti-faux positifs)."""
    s = pd.Series(flags.astype(int))
    return (s.rolling(k, min_periods=k).sum() == k).fillna(False).to_numpy()


def main():
    print("[1/6] Simulation des données...")
    raw = simulate()
    raw.to_csv(OUT / "sensor_data_simulated.csv", index=False)

    print("[2/6] Feature engineering...")
    df = build_features(raw)
    day = 24 * 3600 // SAMPLE_SEC
    d_idx = ((df["timestamp"] - raw["timestamp"].iloc[0]).dt.total_seconds() // 86400).astype(int)
    train = df[d_idx < TRAIN_DAYS]
    val = df[(d_idx >= TRAIN_DAYS) & (d_idx < TRAIN_DAYS + VAL_DAYS)]
    test = df[d_idx >= TRAIN_DAYS + VAL_DAYS].reset_index(drop=True)
    assert (train["label"] == 0).all() and (val["label"] == 0).all()
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    print("[3/6] Entraînement Isolation Forest (données saines)...")
    scaler = StandardScaler().fit(train[FEATURES])
    model = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto",
                            random_state=SEED, n_jobs=-1)
    model.fit(scaler.transform(train[FEATURES]))

    print("[4/6] Calibration du seuil sur la validation saine...")
    val_score = -model.score_samples(scaler.transform(val[FEATURES]))     # + haut = + anormal
    threshold = float(np.quantile(val_score, 1 - FALSE_ALARM_QUANTILE))

    print("[5/6] Évaluation sur le jeu de test...")
    test["anomaly_score"] = -model.score_samples(scaler.transform(test[FEATURES]))
    test["raw_flag"] = test["anomaly_score"] > threshold
    test["ml_alert"] = persistence_filter(test["raw_flag"].to_numpy(), PERSISTENCE)
    # Baseline interdite par le sujet, utilisée ici uniquement comme point de comparaison
    test["static_alert"] = (test["temperature"] > T_CRIT) | (test["gas_ppm"] > GAS_CRIT)

    ev_mask = ~test["recovery"].to_numpy()          # on exclut la phase de retour à la normale
    y = (test["label"] > 0).astype(int).to_numpy()[ev_mask]
    metrics = {"threshold": threshold, "persistence_samples": PERSISTENCE}
    for name, col in (("isolation_forest", "ml_alert"), ("static_threshold", "static_alert")):
        pred = test[col].astype(int).to_numpy()[ev_mask]
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        metrics[name] = {
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
    metrics["isolation_forest"]["roc_auc"] = round(roc_auc_score(y, test["anomaly_score"].to_numpy()[ev_mask]), 4)

    # Niveau incident : détecté ? délai d'anticipation avant le seuil critique physique
    rows = []
    for k in sorted(test.loc[test["event_id"] >= 0, "event_id"].unique()):
        ev = test[test["event_id"] == k]
        etype = ev["event_type"].iloc[0]
        start = ev["timestamp"].iloc[0]
        crit = ev[(ev["temperature"] > T_CRIT) | (ev["gas_ppm"] > GAS_CRIT)]
        t_crit = crit["timestamp"].iloc[0] if len(crit) else pd.NaT
        ml = ev[ev["ml_alert"]]
        t_ml = ml["timestamp"].iloc[0] if len(ml) else pd.NaT
        lead = (t_crit - t_ml).total_seconds() / 60 if pd.notna(t_ml) and pd.notna(t_crit) else np.nan
        rows.append({
            "event_id": int(k), "type": etype, "start": str(start),
            "ml_detected": bool(pd.notna(t_ml)),
            "detection_delay_min": round((t_ml - start).total_seconds() / 60, 1) if pd.notna(t_ml) else None,
            "static_threshold_crossed_at": str(t_crit),
            "lead_time_before_critical_min": None if np.isnan(lead) else round(lead, 1),
        })
    events = pd.DataFrame(rows)
    events.to_csv(OUT / "event_level_results.csv", index=False)

    # Fausses alertes sur périodes saines du test (hors incidents et leur décroissance)
    healthy = (test["event_id"] == -1) & ~test["recovery"]
    # exclure une marge de 40 min après chaque incident (retour à la normale)
    margin = pd.Series(False, index=test.index)
    for k in events["event_id"]:
        ev_idx = test.index[test["event_id"] == k]
        margin.iloc[ev_idx.max(): ev_idx.max() + 40 * PER_MIN] = True
    clean = healthy & ~margin
    alarm_runs = (test["ml_alert"] & clean).astype(int).diff().fillna(0).eq(1).sum()
    metrics["false_alarm_episodes_in_healthy_test_periods"] = int(alarm_runs)
    metrics["events_detected"] = f"{int(events['ml_detected'].sum())}/{len(events)}"
    metrics["mean_lead_time_before_critical_min"] = (
        round(float(events["lead_time_before_critical_min"].dropna().mean()), 1)
        if events["lead_time_before_critical_min"].notna().any() else None)

    # --------------------------------------------------------------------- #
    # 6. Exports
    # --------------------------------------------------------------------- #
    print("[6/6] Export des résultats...")
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES,
                 "threshold": threshold, "persistence": PERSISTENCE},
                OUT / "asphaleia_isolation_forest.joblib")
    test.drop(columns=["raw_flag"]).to_csv(OUT / "test_predictions.csv", index=False)
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- Graphique 1 : vue d'ensemble du test ---
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(test["timestamp"], test["temperature"], lw=.6, color="tab:red")
    ax[0].axhline(T_CRIT, ls="--", color="k", lw=.8, label=f"seuil statique {T_CRIT} °C")
    ax[0].set_ylabel("Température (°C)"); ax[0].legend(loc="upper left")
    ax[1].plot(test["timestamp"], test["gas_ppm"], lw=.6, color="tab:green")
    ax[1].axhline(GAS_CRIT, ls="--", color="k", lw=.8, label=f"seuil statique {GAS_CRIT:.0f} ppm")
    ax[1].set_ylabel("Gaz MQ-2 (ppm)"); ax[1].legend(loc="upper left")
    ax[2].plot(test["timestamp"], test["anomaly_score"], lw=.5, color="tab:purple")
    ax[2].axhline(threshold, color="r", ls="--", label="seuil calibré (validation)")
    ax[2].set_ylabel("Score d'anomalie"); ax[2].legend(loc="upper left")
    for a in ax:
        a.fill_between(test["timestamp"], 0, 1, where=test["label"] > 0,
                       transform=a.get_xaxis_transform(), color="orange", alpha=.25)
    fig.suptitle("ASPHALEIA : jeu de test (zones orange = incidents réels)")
    fig.tight_layout(); fig.savefig(OUT / "fig1_overview_test.png", dpi=130); plt.close(fig)

    # --- Graphique 2 : zoom sur chaque incident ---
    n_ev = len(events)
    fig, axes = plt.subplots(n_ev, 1, figsize=(13, 3.2 * n_ev))
    for a, (_, r) in zip(np.atleast_1d(axes), events.iterrows()):
        ev = test[test["event_id"] == r["event_id"]]
        lo, hi = ev.index.min() - 30 * PER_MIN, ev.index.max() + 10 * PER_MIN
        seg = test.loc[max(lo, 0):hi]
        a.plot(seg["timestamp"], seg["temperature"], color="tab:red", lw=.8, label="Température")
        a.set_ylabel("°C")
        b = a.twinx()
        b.plot(seg["timestamp"], seg["gas_ppm"], color="tab:green", lw=.8, label="Gaz")
        b.set_ylabel("ppm")
        a.fill_between(seg["timestamp"], 0, 1, where=seg["label"] == 1, transform=a.get_xaxis_transform(),
                       color="gold", alpha=.35, label="précurseur")
        a.fill_between(seg["timestamp"], 0, 1, where=seg["label"] == 2, transform=a.get_xaxis_transform(),
                       color="red", alpha=.2, label="critique")
        first = seg[seg["ml_alert"]]
        if len(first):
            a.axvline(first["timestamp"].iloc[0], color="purple", lw=2, label="1re alerte IA")
        crit = seg[(seg["temperature"] > T_CRIT) | (seg["gas_ppm"] > GAS_CRIT)]
        if len(crit):
            a.axvline(crit["timestamp"].iloc[0], color="k", ls="--", label="seuil statique franchi")
        lt = r["lead_time_before_critical_min"]
        a.set_title(f"Incident {r['event_id']} ({r['type']}) : anticipation = "
                    f"{'n/a' if lt is None or pd.isna(lt) else f'{lt} min'}")
        a.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig2_events_zoom.png", dpi=130); plt.close(fig)

    # --- Graphique 3 : ROC + distribution des scores ---
    fpr, tpr, _ = roc_curve(y, test["anomaly_score"].to_numpy()[ev_mask])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(fpr, tpr, label=f"AUC = {metrics['isolation_forest']['roc_auc']}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
    ax[0].set_xlabel("Taux de faux positifs"); ax[0].set_ylabel("Taux de vrais positifs")
    ax[0].set_title("Courbe ROC"); ax[0].legend()
    ax[1].hist(val_score, bins=80, alpha=.6, density=True, label="validation (sain)")
    ax[1].hist(test.loc[test["label"] > 0, "anomaly_score"], bins=80, alpha=.6, density=True, label="test : incidents")
    ax[1].axvline(threshold, color="r", ls="--", label="seuil")
    ax[1].set_xlabel("Score d'anomalie"); ax[1].set_title("Distribution des scores"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "fig3_roc_scores.png", dpi=130); plt.close(fig)

    # --- Rapport Markdown ---
    m1, m2 = metrics["isolation_forest"], metrics["static_threshold"]
    report = f"""# ASPHALEIA : Résultats maintenance prédictive (Isolation Forest)

## Données simulées
- {N_DAYS} jours, 1 mesure / {SAMPLE_SEC} s (température DHT22, gaz MQ-2), {N} points
- Train : jours 0-{TRAIN_DAYS} (sain) | Validation : jours {TRAIN_DAYS}-{TRAIN_DAYS + VAL_DAYS} (sain) | Test : jours {TRAIN_DAYS + VAL_DAYS}-{N_DAYS} avec 6 incidents (3 dérives lentes, 2 fuites de gaz, 1 surchauffe)
- {len(FEATURES)} features : valeurs brutes, moyennes/écarts-types 5 min, pentes 15 min, écart à la moyenne 1 h, corrélation température/gaz 15 min

## Modèle
- IsolationForest (300 arbres), entraîné uniquement sur données saines (approche non supervisée)
- Seuil = quantile {1 - FALSE_ALARM_QUANTILE:.3f} des scores de validation saine ({threshold:.4f})
- Filtre de persistance : {PERSISTENCE} détections consécutives (~1 min)

## Résultats (niveau échantillon, jeu de test)
| Méthode | Précision | Rappel | F1 | FP | FN |
|---|---|---|---|---|---|
| Isolation Forest | {m1['precision']} | {m1['recall']} | {m1['f1']} | {m1['fp']} | {m1['fn']} |
| Seuil statique (baseline) | {m2['precision']} | {m2['recall']} | {m2['f1']} | {m2['fp']} | {m2['fn']} |

ROC-AUC (Isolation Forest) : {m1['roc_auc']}

## Résultats (niveau incident)
- Incidents détectés : {metrics['events_detected']}
- Anticipation moyenne avant franchissement du seuil critique : {metrics['mean_lead_time_before_critical_min']} min
- Épisodes de fausses alertes en période saine : {metrics['false_alarm_episodes_in_healthy_test_periods']}

{events.to_markdown(index=False)}
"""
    (OUT / "results_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()






