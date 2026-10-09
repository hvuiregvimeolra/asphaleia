# ASPHALEIA

Projet de supervision : dashboard + API + Isolation Forest + webcam YuNet/SFace + MQTT.

## Démarrage
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python3 asphaleia_predictive_maintenance.py` (une fois pour créer le modèle)
5. `docker compose up -d`
6. `python3 asphaleia_api.py`
7. Dans un autre terminal : `python3 simulateur.py`
8. Ouvrir http://localhost:8000

## Caméra
1. Enregistrer une personne : `python3 face.py enroll Dihia`
2. Lancer la surveillance : `python3 face.py run --api http://localhost:8000`
3. Dans le dashboard, ouvrir l'onglet Caméra.

La caméra télécharge automatiquement les modèles YuNet et SFace dans `face_models/` s'ils sont absents.
