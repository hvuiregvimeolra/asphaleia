#!/usr/bin/env python3
"""
Simulateur ASPHALEIA.
Il remplace l'ESP8266 et les capteurs physiques : chaque seconde il invente une
température, un taux de gaz et une présence, puis les envoie.

Deux façons d'envoyer :
  python3 simulateur.py          -> par MQTT (il faut le broker : docker compose up -d)
  python3 simulateur.py --http   -> directement à l'API, sans broker (plus simple pour tester)

Pendant qu'il tourne, tape un mot puis Entrée pour lancer un scénario :
  incendie | gaz | intrusion | pic | normal | quitter
"""
import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request

DUREES = {"incendie": 30, "gaz": 30, "intrusion": 20}   # en secondes


class Capteurs:
    """Les trois capteurs virtuels."""

    def __init__(self):
        self.t = 0
        self.temp = 24.0
        self.gaz = 120.0
        self.presence = False
        self.scenario = None
        self.restant = 0

    def lancer(self, nom):
        if nom == "normal":
            self.scenario, self.restant = None, 0
        elif nom == "pic":
            self.temp += 14
        elif nom in DUREES:
            self.scenario, self.restant = nom, DUREES[nom]
        else:
            print(f"Scénario inconnu : {nom}")
            return
        print(f"-> scénario : {nom}")

    def seconde(self):
        """Fait avancer d'une seconde et renvoie (température, gaz, présence)."""
        self.t += 1
        cible_t = 24 + math.sin(self.t / 30) * 0.8      # petites variations normales
        cible_g = 120 + math.sin(self.t / 20) * 8
        presence = False
        if self.scenario == "incendie":
            cible_t, cible_g = 62, 520
        elif self.scenario == "gaz":
            cible_t, cible_g = 26, 950
        elif self.scenario == "intrusion":
            presence = True
        if self.scenario:
            self.restant -= 1
            if self.restant <= 0:
                self.scenario = None
                print("-> scénario terminé, retour à la normale")
        self.temp += (cible_t - self.temp) * 0.12 + random.uniform(-0.3, 0.3)
        self.gaz = max(0.0, self.gaz + (cible_g - self.gaz) * 0.12 + random.uniform(-4, 4))
        self.presence = presence
        return round(self.temp, 1), round(self.gaz), presence


def transport_mqtt(args, capteurs):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("Il manque paho-mqtt : pip install -r requirements.txt")

    def a_la_connexion(client, userdata, flags, reason_code, properties):
        print(f"Connecté au broker MQTT ({args.host}:{args.port})")
        client.subscribe("asphaleia/commande")

    def a_un_message(client, userdata, msg):
        try:
            capteurs.lancer(json.loads(msg.payload)["scenario"])
        except (ValueError, KeyError):
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = a_la_connexion
    client.on_message = a_un_message
    try:
        client.connect(args.host, args.port)
    except OSError:
        sys.exit(f"Impossible de joindre le broker sur {args.host}:{args.port}. "
                 "Lance-le avec : docker compose up -d   (ou essaie : python3 simulateur.py --http)")
    client.loop_start()

    def envoyer(temp, gaz, presence):
        client.publish("asphaleia/capteurs/temperature", json.dumps({"value": temp, "unit": "C"}))
        client.publish("asphaleia/capteurs/gaz", json.dumps({"value": gaz, "unit": "ppm"}))
        client.publish("asphaleia/capteurs/presence", json.dumps({"value": presence}))
    return envoyer, lambda: None


def transport_http(args, capteurs):
    base = args.api.rstrip("/")

    def appel(chemin, donnees=None):
        req = urllib.request.Request(base + chemin, method="POST" if donnees is not None else "GET")
        if donnees is not None:
            req.add_header("Content-Type", "application/json")
            donnees = json.dumps(donnees).encode()
        with urllib.request.urlopen(req, data=donnees, timeout=3) as r:
            return json.loads(r.read() or b"{}")

    def envoyer(temp, gaz, presence):
        try:
            appel("/api/mesures", {"temperature": temp, "gaz": gaz, "presence": presence})
        except (urllib.error.URLError, OSError):
            print(f"API injoignable sur {base} : est-elle lancée ? (python3 api.py)")

    def relever_commandes():
        try:
            nom = appel("/api/commande").get("scenario")
            if nom:
                capteurs.lancer(nom)
        except (urllib.error.URLError, OSError):
            pass
    return envoyer, relever_commandes


def lire_clavier(capteurs, arret):
    for ligne in sys.stdin:
        mot = ligne.strip().lower()
        if mot in ("quitter", "q", "exit"):
            arret.set()
            return
        if mot:
            capteurs.lancer(mot)


def main():
    p = argparse.ArgumentParser(description="Simulateur de capteurs ASPHALEIA")
    p.add_argument("--http", action="store_true", help="envoyer à l'API sans passer par MQTT")
    p.add_argument("--api", default="http://localhost:8000", help="adresse de l'API (mode --http)")
    p.add_argument("--host", default="localhost", help="adresse du broker MQTT")
    p.add_argument("--port", type=int, default=1883, help="port du broker MQTT")
    args = p.parse_args()

    capteurs = Capteurs()
    envoyer, relever = (transport_http if args.http else transport_mqtt)(args, capteurs)

    arret = threading.Event()
    threading.Thread(target=lire_clavier, args=(capteurs, arret), daemon=True).start()
    print("Simulateur lancé. Scénarios : incendie | gaz | intrusion | pic | normal | quitter")

    try:
        while not arret.is_set():
            debut = time.time()
            relever()
            temp, gaz, presence = capteurs.seconde()
            envoyer(temp, gaz, presence)
            print(f"{time.strftime('%H:%M:%S')}  {temp:5.1f} °C   {gaz:4d} ppm   présence : {'oui' if presence else 'non'}")
            time.sleep(max(0, 1 - (time.time() - debut)))
    except KeyboardInterrupt:
        pass
    print("Simulateur arrêté.")


if __name__ == "__main__":
    main()
