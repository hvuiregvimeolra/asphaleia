#!/usr/bin/env python3
"""
ASPHALEIA : API qui relie le dashboard aux deux modèles IA
===========================================================
  - Isolation Forest : charge le modèle entraîné par asphaleia_predictive_maintenance.py
    (fichier .joblib), calcule les mêmes 11 variables en fenêtres glissantes, le score
    d'anomalie, applique le seuil calibré et le filtre de persistance (6 détections).
  - Reconnaissance faciale : reçoit les alertes et l'état de face.py
    (python face.py run --api http://localhost:8000).
  - Dashboard : sert asphaleia.html et répond à ses requêtes (http://localhost:8000).

Reprise du serveur de l'archive ASPHALEA (api.py) :
  - MQTT (optionnel, pip install paho-mqtt) : écoute asphaleia/capteurs/# et publie
    asphaleia/commande ; sans broker, l'API démarre quand même.
  - Base SQLite asphaleia.db : mesures + alertes enregistrées (même schéma que l'archive).
  - POST /api/mesures {"temperature","gaz","presence"} (simulateur.py --http),
    POST /api/vision {"personnes","confiance"} (caméra YOLO), GET /api/commande,
    GET /api/historique, GET /api/alertes, POST /api/alertes/ack-tout, /api/alertes/<id>/ack.

Capteurs : par défaut un simulateur (même profil que l'entraînement : ~25 °C, ~180 ppm).
Pour utiliser de vraies mesures (ESP8266, Wokwi...), envoyez-les en POST JSON sur
/api/sensors : {"temperature": 25.3, "gas_ppm": 181, "presence": false}.
Dès que des mesures réelles arrivent (moins de 5 s), elles remplacent le simulateur.

Le modèle a été entraîné avec 1 mesure toutes les 10 s. Pour que la démo réagisse
vite, 1 tick de l'API (1 s) représente 1 mesure : le temps est accéléré x10.

Usage :
    pip install numpy pandas scikit-learn joblib
    python asphaleia_predictive_maintenance.py      # une fois : entraîne et exporte le modèle
    python asphaleia_api.py                         # puis ouvrir http://localhost:8000
"""
import argparse
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import paho.mqtt.client as mqtt
except ImportError:      # l'API marche sans MQTT
    mqtt = None

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "outputs_asphaleia" / "asphaleia_isolation_forest.joblib"
DEFAULT_HTML = BASE / "asphaleia.html"
DEFAULT_DB = os.getenv("ASPHALEIA_DB", str(BASE / "asphaleia.db"))
TOPIC_CAPTEURS = "asphaleia/capteurs/"
TOPIC_COMMANDE = "asphaleia/commande"

# Mêmes constantes que le script d'entraînement (1 mesure / 10 s)
PER_MIN = 6
W5, W15, W60 = 5 * PER_MIN, 15 * PER_MIN, 60 * PER_MIN
SAMPLE_SEC = 10          # temps simulé représenté par un tick
BUF = 470                # il faut au moins 60 min + 5 min de mesures pour calculer les variables
HIST = 60                # points renvoyés au dashboard
EXT_TTL = 5.0            # s : durée pendant laquelle une mesure réelle reste valable
ALERT_COOLDOWN = 12      # ticks entre deux alertes IA
MAX_BODY = 3_000_000


# --------------------------------------------------------------------------- #
# Variables du modèle : copie exacte de build_features() du script d'entraînement
# --------------------------------------------------------------------------- #
def build_features(temp, gas):
    d = pd.DataFrame({"temperature": np.asarray(temp, float), "gas_ppm": np.asarray(gas, float)})
    for c, p in (("temperature", "temp"), ("gas_ppm", "gas")):
        d[f"{p}_mean_5m"] = d[c].rolling(W5, min_periods=W5).mean()
        d[f"{p}_std_5m"] = d[c].rolling(W5, min_periods=W5).std()
        d[f"{p}_slope_15m"] = (d[f"{p}_mean_5m"] - d[f"{p}_mean_5m"].shift(W15)) / 15
        d[f"{p}_dev_1h"] = d[f"{p}_mean_5m"] - d[c].rolling(W60, min_periods=W60).mean().shift(W5)
    d["corr_temp_gas_15m"] = d["temperature"].rolling(W15, min_periods=W15).corr(d["gas_ppm"])
    return d


# --------------------------------------------------------------------------- #
# Capteurs simulés
# --------------------------------------------------------------------------- #
class Source:
    def __init__(self):
        self.rng = np.random.default_rng()
        lt = time.localtime()
        self.hour = lt.tm_hour + lt.tm_min / 60
        self.ar_t = self.ar_g = 0.0
        self.t_off = self.g_off = 0.0
        self.scen, self.left, self.dirty = None, 0, False

    def _noise(self):
        self.ar_t = 0.995 * self.ar_t + self.rng.normal(0, 0.03)
        self.ar_g = 0.98 * self.ar_g + self.rng.normal(0, 0.8)

    def _base(self, hour):
        temp = 25 + 3.0 * math.sin(2 * math.pi * (hour - 9) / 24) + self.ar_t + self.rng.normal(0, 0.15)
        gas = 180 + 1.5 * (temp - 25) + self.ar_g + self.rng.normal(0, 3.0)
        return temp, gas

    def healthy_history(self, n):
        """n mesures saines qui se terminent à l'heure courante."""
        ts, gs = [], []
        start = self.hour - n * SAMPLE_SEC / 3600
        for i in range(n):
            self._noise()
            t, g = self._base((start + i * SAMPLE_SEC / 3600) % 24)
            ts.append(t)
            gs.append(g)
        return ts, gs

    def step(self):
        self._noise()
        temp, gas = self._base(self.hour)
        self.hour = (self.hour + SAMPLE_SEC / 3600) % 24
        tt, tg, pres = 0.0, 0.0, False
        if self.scen == "incendie":
            tt, tg = 37.0, 340.0
        elif self.scen == "gaz":
            tt, tg = 1.0, 770.0
        elif self.scen == "intrusion":
            pres = True
        self.t_off += (tt - self.t_off) * 0.12
        self.g_off += (tg - self.g_off) * 0.12
        if self.scen:
            self.left -= 1
            if self.left <= 0:
                self.scen = None
        return temp + self.t_off, gas + self.g_off, pres

    def settled(self):
        return self.scen is None and abs(self.t_off) < 0.5 and abs(self.g_off) < 5


SCENARIOS = {"incendie": 30, "gaz": 30, "intrusion": 20}


# --------------------------------------------------------------------------- #
# Moteur : modèle + historique + alertes
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, model_path, db_path=DEFAULT_DB, mqtt_host=None, mqtt_port=1883):
        art = joblib.load(model_path)
        self.model, self.scaler = art["model"], art["scaler"]
        self.features = list(art["features"])
        self.calibrated = float(art["threshold"])
        self.thr = self.calibrated
        self.persistence = int(art.get("persistence", 6))
        self.tidx = [i for i, f in enumerate(self.features) if f == "temperature" or f.startswith("temp_")]
        self.gidx = [i for i, f in enumerate(self.features) if f == "gas_ppm" or f.startswith("gas_")]
        self.lock = threading.Lock()
        self.boot = int(time.time())
        self.src = Source()
        self.ext, self.ext_t = None, 0.0
        self.mode = "simulation"
        self.alerts, self.next_id = [], 1
        self.cool = 0
        self.face, self.face_mono = None, 0.0
        self.face_alert_mono = 0.0
        self.last = None
        self.vision_ts, self.vision_hits, self.vision_pause = 0.0, 0, 0.0
        self.pending_cmd = None          # commande en attente pour simulateur.py --http
        self.mqtt_ok, self.mqtt_client = False, None
        self._open_db(db_path)
        if mqtt_host:
            self._start_mqtt(mqtt_host, mqtt_port)
        self._reset_history()
        self._compute(append=False)

    # ---- base de données (SQLite, schéma de l'archive) ------------------ #
    def _open_db(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS mesures (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, temperature REAL,
            gaz REAL, presence INTEGER, score REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS alertes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, heure TEXT, source TEXT,
            severite TEXT, titre TEXT, detail TEXT, acquittee INTEGER DEFAULT 0)""")
        self.db.commit()
        row = self.db.execute("SELECT COALESCE(MAX(id),0) FROM alertes").fetchone()
        self.next_id = int(row[0]) + 1       # les ids restent uniques d'un lancement à l'autre

    def _save_sample(self, t, g, p, score):
        self.db.execute("INSERT INTO mesures (ts, temperature, gaz, presence, score) VALUES (?,?,?,?,?)",
                        (time.time(), float(t), float(g), int(bool(p)), float(score)))
        self.db.commit()

    def history_rows(self, n):
        with self.lock:
            rows = self.db.execute("SELECT ts, temperature, gaz, presence, score FROM mesures "
                                   "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"ts": r[0], "temperature": r[1], "gaz": r[2], "presence": bool(r[3]), "score": r[4]}
                for r in reversed(rows)]

    def alert_rows(self, n):
        with self.lock:
            rows = self.db.execute("SELECT id, heure, source, severite, titre, detail, acquittee FROM alertes "
                                   "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"id": r[0], "heure": r[1], "source": r[2], "severite": r[3], "titre": r[4],
                 "detail": r[5], "acquittee": bool(r[6])} for r in rows]

    def ack(self, alert_id=None):
        with self.lock:
            if alert_id is None:
                self.db.execute("UPDATE alertes SET acquittee = 1")
            else:
                self.db.execute("UPDATE alertes SET acquittee = 1 WHERE id = ?", (int(alert_id),))
            self.db.commit()
            for a in self.alerts:
                if alert_id is None or a["id"] == int(alert_id):
                    a["ack"] = True

    # ---- MQTT (optionnel) ---------------------------------------------- #
    def _start_mqtt(self, host, port):
        if mqtt is None:
            print("[API] paho-mqtt non installé : MQTT désactivé (POST /api/mesures reste possible).")
            return

        def on_connect(client, userdata, flags, reason_code, properties):
            self.mqtt_ok = True
            client.subscribe(TOPIC_CAPTEURS + "#")
            print(f"[API] MQTT connecté ({host}:{port})")

        def on_disconnect(client, userdata, flags, reason_code, properties):
            self.mqtt_ok = False
            print("[API] MQTT déconnecté, nouvelle tentative automatique...")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect, client.on_disconnect = on_connect, on_disconnect
        client.on_message = lambda c, u, m: self.on_mqtt(m.topic, m.payload)
        client.connect_async(host, port)        # réessaie seul si le broker démarre plus tard
        client.loop_start()
        self.mqtt_client = client

    def on_mqtt(self, topic, payload):
        """Un message sur asphaleia/capteurs/<temperature|gaz|presence> : {"value": 24.1} ou 24.1."""
        if not topic.startswith(TOPIC_CAPTEURS):
            return
        try:
            data = json.loads(payload)
            self.set_partial({topic[len(TOPIC_CAPTEURS):]: data["value"] if isinstance(data, dict) else data})
        except (ValueError, KeyError, TypeError):
            print(f"[API] message MQTT ignoré sur {topic} : {payload!r}")

    # ---- historique ---------------------------------------------------- #
    def _reset_history(self):
        ts, gs = self.src.healthy_history(BUF)
        self.temp, self.gas = deque(ts, maxlen=BUF), deque(gs, maxlen=BUF)
        self.pres = deque([False] * BUF, maxlen=BUF)
        self.cool = 0

    def _compute(self, append=True, sample=None):
        if append:
            t, g, p = sample
            self.temp.append(t)
            self.gas.append(g)
            self.pres.append(bool(p))
        df = build_features(self.temp, self.gas)
        feat = df.dropna().tail(HIST)
        xs = self.scaler.transform(feat[self.features])
        scores = -self.model.score_samples(xs)
        z = np.abs(xs)
        flags = scores > self.thr
        tz, gz = z[:, self.tidx].max(axis=1), z[:, self.gidx].max(axis=1)
        pres = list(self.pres)[-len(feat):]
        self.last = {
            "t": feat["temperature"].to_numpy(), "g": feat["gas_ppm"].to_numpy(),
            "s": scores, "flags": flags, "tz": tz, "gz": gz, "p": pres,
        }

    # ---- un tick ------------------------------------------------------- #
    def tick(self):
        with self.lock:
            now = time.monotonic()
            sim = self.src.step()
            if self.ext is not None and now - self.ext_t < EXT_TTL:
                sample = (self.ext["temperature"], self.ext["gas_ppm"], self.ext["presence"])
                self.mode = "capteurs"
            else:
                sample = sim
                self.mode = "simulation"
                if self.src.dirty and self.src.settled():
                    self.src.dirty = False
                    self._reset_history()   # démo : le retour à la normale efface l'historique perturbé
            self._compute(sample=sample)
            L = self.last
            try:
                self._save_sample(sample[0], sample[1], sample[2], L["s"][-1])
            except sqlite3.Error as e:
                print("[API] erreur base :", e)
            if self.cool > 0:
                self.cool -= 1
            tail = L["flags"][-self.persistence:]
            if len(tail) == self.persistence and bool(tail.all()) and self.cool == 0:
                self.cool = ALERT_COOLDOWN
                self._raise_ia()

    def _ratios(self):
        L = self.last
        # écart type le plus fort parmi les variables de chaque capteur, en échelle log (les écarts vont de 1 à 400)
        fr = lambda z: min(1.0, math.log1p(float(z)) / math.log1p(400.0))
        rt, rg = (5 * fr(L["tz"][-1])) ** 2, (5 * fr(L["gz"][-1])) ** 2
        face_hot = time.monotonic() - self.face_alert_mono < 10
        rp = 9.0 if face_hot else 0.0
        s = (rt + rg + rp) or 1.0
        return {"t": rt, "g": rg, "p": rp}, {"t": rt / s, "g": rg / s, "p": rp / s}

    def _raise_ia(self):
        raw, comp = self._ratios()
        L = self.last
        score = float(L["s"][-1])
        if comp["g"] >= comp["t"]:
            title, detail = "Concentration de gaz anormale", f"{round(float(L['g'][-1]))} ppm"
        else:
            title, detail = "Température anormale", f"{float(L['t'][-1]):.1f} °C"
        sev = "crit" if score > self.thr + 0.12 else "warn"
        self._add_alert("ia", sev, title, f"{detail} · score {score:.2f}")

    def _add_alert(self, src, sev, title, detail):
        a = {"id": self.next_id, "src": src, "sev": sev, "title": title[:80],
             "detail": detail[:200], "time": time.strftime("%H:%M:%S"), "ack": False}
        self.next_id += 1
        self.alerts.append(a)
        del self.alerts[:-60]
        try:
            self.db.execute("INSERT INTO alertes (id, ts, heure, source, severite, titre, detail) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (a["id"], time.time(), a["time"], src, sev, a["title"], a["detail"]))
            self.db.commit()
        except sqlite3.Error as e:
            print("[API] erreur base :", e)
        print(f"[ALERTE {sev.upper()}] {src} : {a['title']} ({a['detail']})")
        return a

    # ---- entrées ------------------------------------------------------- #
    def set_sensors(self, d):
        with self.lock:
            self.ext = {"temperature": float(d["temperature"]), "gas_ppm": float(d["gas_ppm"]),
                        "presence": bool(d.get("presence", False))}
            self.ext_t = time.monotonic()

    def set_partial(self, d):
        """Mesures reçues une par une (MQTT) ou en vrac (/api/mesures). Accepte gaz ou gas_ppm."""
        with self.lock:
            cur = dict(self.ext) if self.ext is not None else {
                "temperature": float(self.temp[-1]), "gas_ppm": float(self.gas[-1]), "presence": False}
            if "temperature" in d:
                cur["temperature"] = float(d["temperature"])
            for k in ("gaz", "gas_ppm"):
                if k in d:
                    cur["gas_ppm"] = max(0.0, float(d[k]))
            if "presence" in d:
                cur["presence"] = bool(d["presence"])
            self.ext, self.ext_t = cur, time.monotonic()

    def vision(self, d):
        """Compatibilité caméra YOLO de l'archive : {"personnes": 1, "confiance": 0.91}."""
        persons, conf = int(d.get("personnes", 0)), float(d.get("confiance", 0))
        with self.lock:
            now = time.monotonic()
            self.vision_ts = now
            self.vision_hits = self.vision_hits + 1 if persons >= 1 else 0
            fire = self.vision_hits >= 2 and now >= self.vision_pause
            if fire:
                self.vision_pause = now + 15
                self.face_alert_mono = now
                return self._add_alert("yolo", "crit", "Intrusion détectée", f"personne · confiance {conf:.2f}")
        return None

    def command(self, name):
        """Lance un scénario : MQTT si le broker est là, sinon file d'attente pour simulateur.py --http."""
        self.set_scenario(name)
        if self.mqtt_ok and self.mqtt_client:
            self.mqtt_client.publish(TOPIC_COMMANDE, json.dumps({"scenario": name}))
            return "mqtt"
        self.pending_cmd = name
        return "http"

    def take_command(self):
        name, self.pending_cmd = self.pending_cmd, None
        return name

    def set_scenario(self, name):
        with self.lock:
            if name in SCENARIOS:
                self.src.scen, self.src.left, self.src.dirty = name, SCENARIOS[name], True
            elif name == "spike":
                self.src.t_off += 14.0
                self.src.dirty = True
            elif name == "pic":
                self.src.t_off += 14.0
                self.src.dirty = True
            elif name in ("calm", "normal"):
                self.src.scen, self.src.left = None, 0
                self.src.t_off = self.src.g_off = 0.0
                self.src.dirty = False
                self._reset_history()
            else:
                raise ValueError("scénario inconnu")

    def set_threshold(self, v):
        with self.lock:
            self.thr = min(0.95, max(0.30, float(v)))
            self._compute(append=False)

    def add_face_alert(self, p):
        kind = str(p.get("type", ""))[:40]
        snap = str(p.get("snapshot", ""))[:80]
        sev = "crit" if str(p.get("severity", "high")).lower() in ("high", "critical", "crit") else "warn"
        if kind == "spoofing":
            title = "Spoofing détecté"
            detail = f"test de vivacité échoué · défi {str(p.get('challenge', '?'))[:10]}"
        elif kind == "unknown_face":
            title, detail = "Intrusion détectée", "visage inconnu · liveness OK"
        else:
            title = (kind or "Alerte vision").replace("_", " ").capitalize()
            detail = str(p.get("message", "alerte envoyée par face.py"))
        if snap:
            detail += f" · {snap}"
        with self.lock:
            self.face_alert_mono = time.monotonic()
            return self._add_alert("face", sev, title, detail)

    def set_face(self, d):
        faces = []
        for f in (d.get("faces") or [])[:8]:
            box = [int(v) for v in (f.get("box") or [0, 0, 0, 0])[:4]]
            faces.append({"box": box, "label": str(f.get("label", ""))[:40],
                          "kind": f.get("kind") if f.get("kind") in ("known", "unknown", "pending") else "unknown",
                          "score": float(f.get("score", 0) or 0)})
        frame = d.get("frame")
        face = {
            "w": int(d.get("w", 640)), "h": int(d.get("h", 480)), "faces": faces,
            "liveness": d.get("liveness") if d.get("liveness") in ("ok", "failed", "pending", "waiting") else "waiting",
            "message": str(d.get("message", ""))[:80], "intrusion": bool(d.get("intrusion", False)),
            "frame": frame if isinstance(frame, str) and len(frame) < 2_500_000 else None,
        }
        with self.lock:
            self.face, self.face_mono = face, time.monotonic()

    # ---- sorties ------------------------------------------------------- #
    def _face_summary(self):
        f = self.face
        if not f or time.monotonic() - self.face_mono > 3.0:
            return {"fresh": False, "status": "HORS LIGNE", "identity": None, "conf": 0.0,
                    "liveness": "waiting", "message": "", "n": 0}
        known = [x for x in f["faces"] if x["kind"] == "known"]
        unknown = [x for x in f["faces"] if x["kind"] == "unknown"]
        best = max(known, key=lambda x: x["score"], default=None)
        if f["liveness"] == "failed":
            status = "SPOOFING"
        elif unknown or f["intrusion"]:
            status = "INTRUS"
        elif f["liveness"] == "ok" and best:
            status = "AUTORISÉE"
        elif f["liveness"] == "pending":
            status = "VÉRIFICATION"
        else:
            status = "AUCUN VISAGE"
        name = best["label"].rsplit(" ", 1)[0] if best else None
        return {"fresh": True, "status": status, "identity": name, "conf": best["score"] if best else 0.0,
                "liveness": f["liveness"], "message": f["message"], "n": len(f["faces"])}

    def state(self):
        with self.lock:
            L = self.last
            raw, comp = self._ratios()
            fs = self._face_summary()
            presence = bool(L["p"][-1]) or (fs["fresh"] and fs["n"] > 0)
            hist = []
            for i in range(len(L["s"])):
                fl = bool(L["flags"][i])
                hist.append({"t": round(float(L["t"][i]), 2), "g": round(float(L["g"][i]), 1),
                             "p": bool(L["p"][i]), "s": round(float(L["s"][i]), 4),
                             "ft": fl and bool(L["tz"][i] > 3), "fg": fl and bool(L["gz"][i] > 3)})
            return {
                "boot": self.boot, "mode": self.mode,
                "temp": float(L["t"][-1]), "gas": float(L["g"][-1]), "presence": presence,
                "score": float(L["s"][-1]), "threshold": self.thr, "calibrated": self.calibrated,
                "raw": raw, "comp": comp, "hist": hist,
                "scenario": {"name": self.src.scen, "left": self.src.left},
                "mqtt": self.mqtt_ok, "model": {"loaded": True, "features": len(self.features), "persistence": self.persistence},
                "face": fs, "alerts": list(self.alerts[-30:]),
            }

    def face_frame(self):
        with self.lock:
            fs = self._face_summary()
            f = self.face if fs["fresh"] else None
            out = {"fresh": fs["fresh"], "summary": fs}
            if f:
                out.update({"w": f["w"], "h": f["h"], "faces": f["faces"], "frame": f["frame"]})
            return out


ENGINE = None
HTML_PATH = DEFAULT_HTML


class Handler(BaseHTTPRequestHandler):
    server_version = "ASPHALEIA/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def _n(self, default, maxi):
        m = re.search(r"[?&]n=(\d+)", self.path)
        return max(1, min(maxi, int(m.group(1)))) if m else default

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/asphaleia.html"):
            try:
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._send(404, {"error": f"{HTML_PATH.name} introuvable à côté de asphaleia_api.py"})
        elif path == "/api/state":
            self._send(200, ENGINE.state())
        elif path == "/api/face":
            self._send(200, ENGINE.face_frame())
        elif path == "/api/historique":
            self._send(200, ENGINE.history_rows(self._n(60, 1000)))
        elif path == "/api/alertes":
            self._send(200, ENGINE.alert_rows(self._n(50, 500)))
        elif path == "/api/commande":
            self._send(200, {"scenario": ENGINE.take_command()})
        elif path == "/api/health":
            self._send(200, {"ok": True, "boot": ENGINE.boot})
        else:
            self._send(404, {"error": "introuvable"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY:
                return self._send(413, {"error": "corps trop gros"})
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("objet JSON attendu")
            if path == "/api/alerts":
                a = ENGINE.add_face_alert(data)
                self._send(200, {"ok": True, "id": a["id"]})
            elif path == "/api/face/status":
                ENGINE.set_face(data)
                self._send(200, {"ok": True})
            elif path == "/api/sensors":
                ENGINE.set_sensors(data)
                self._send(200, {"ok": True})
            elif path == "/api/scenario":
                name = str(data.get("name", data.get("nom", "")))
                self._send(200, {"ok": True, "transport": ENGINE.command(name)})
            elif path == "/api/mesures":
                ENGINE.set_partial(data)
                self._send(200, {"ok": True})
            elif path == "/api/vision":
                ENGINE.vision(data)
                self._send(200, {"ok": True})
            elif path == "/api/alertes/ack-tout":
                ENGINE.ack()
                self._send(200, {"ok": True})
            elif (m := re.fullmatch(r"/api/alertes/(\d+)/ack", path)):
                ENGINE.ack(int(m.group(1)))
                self._send(200, {"ok": True})
            elif path in ("/api/threshold", "/api/seuil"):
                ENGINE.set_threshold(data["value"] if "value" in data else data["seuil"])
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "introuvable"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self._send(400, {"error": str(e)})


def main():
    global ENGINE, HTML_PATH
    ap = argparse.ArgumentParser(description="ASPHALEIA : API dashboard + Isolation Forest + face.py")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour ouvrir l'API au réseau local")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--html", default=str(DEFAULT_HTML))
    ap.add_argument("--db", default=DEFAULT_DB, help="fichier SQLite (mesures + alertes)")
    ap.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "localhost"), help="broker MQTT (ex : localhost)")
    ap.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    ap.add_argument("--tick", type=float, default=1.0, help="secondes entre deux mesures")
    args = ap.parse_args()
    HTML_PATH = Path(args.html)
    if not Path(args.model).exists():
        sys.exit(f"Modèle introuvable : {args.model}\n"
                 "Lancez d'abord : python asphaleia_predictive_maintenance.py")
    ENGINE = Engine(args.model, args.db, args.mqtt_host, args.mqtt_port)

    def loop():
        nxt = time.monotonic()
        while True:
            nxt += args.tick
            try:
                ENGINE.tick()
            except Exception as e:      # le dashboard doit continuer à répondre
                print("[API] erreur tick :", e)
            time.sleep(max(0.0, nxt - time.monotonic()))

    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 60)
    print("ASPHALEIA : API démarrée")
    print(f"  Dashboard : http://localhost:{args.port}")
    print(f"  Modèle    : {Path(args.model).name} (seuil calibré {ENGINE.calibrated:.4f}, "
          f"persistance {ENGINE.persistence})")
    print(f"  Base      : {Path(args.db).name}   MQTT : {args.mqtt_host or 'désactivé (--mqtt-host localhost pour l activer)'}")
    print(f"  face.py   : python face.py run --api http://localhost:{args.port}")
    print("  Ctrl+C pour arrêter")
    print("=" * 60)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt...")


if __name__ == "__main__":
    main()
