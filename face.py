#!/usr/bin/env python3

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

MODELS = BASE / "face_models"
DB_DIR = BASE / "faces_db"
DB_FILE = DB_DIR / "faces.json"
OUT = BASE / "face_outputs"

YUNET = (
    "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

SFACE = (
    "face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

W, H = 640, 480

# Reconnaissance
COSINE_THRESHOLD = 0.45

# Détection
FACE_SCORE_THRESHOLD = 0.90

# Temps maximum pour réaliser le mouvement
LIVENESS_TIMEOUT = 5.0

# Nombre de frames nécessaires pour confirmer le mouvement
LIVENESS_CONFIRM_FRAMES = 4

# Nombre de frames avant de confirmer un visage inconnu
CONFIRM_FRAMES = 5

# Temps entre deux alertes
COOLDOWN = 10.0

# Durée d'affichage de l'alerte spoofing
ALERT_DISPLAY_TIME = 3.0


# ============================================================
# MODELES
# ============================================================

def ensure_model(name, url):
    MODELS.mkdir(exist_ok=True)

    path = MODELS / name

    if path.exists() and path.stat().st_size > 100_000:
        return str(path)

    print(f"Téléchargement de {name}...")

    try:
        urllib.request.urlretrieve(url, path)

    except Exception as e:
        sys.exit(
            f"Impossible de télécharger {name} : {e}\n"
            f"Télécharge-le depuis : {url}"
        )

    if path.stat().st_size < 100_000:
        path.unlink(missing_ok=True)

        sys.exit(
            f"Le fichier {name} semble invalide."
        )

    return str(path)


# ============================================================
# MOTEUR DE RECONNAISSANCE
# ============================================================

class FaceEngine:

    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(
            ensure_model(*YUNET),
            "",
            (W, H),
            FACE_SCORE_THRESHOLD,
            0.3,
            5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            ensure_model(*SFACE),
            ""
        )

    def detect(self, frame):

        self.detector.setInputSize(
            (
                frame.shape[1],
                frame.shape[0]
            )
        )

        _, faces = self.detector.detect(frame)

        if faces is None:
            return np.empty(
                (0, 15),
                dtype=np.float32
            )

        return faces

    def embed(self, frame, face):

        aligned = self.recognizer.alignCrop(
            frame,
            face
        )

        return self.recognizer.feature(
            aligned
        ).flatten()


# ============================================================
# BASE DE DONNEES
# ============================================================

def load_db():

    if not DB_FILE.exists():
        return {}

    try:

        data = json.loads(
            DB_FILE.read_text(
                encoding="utf-8"
            )
        )

        return {
            name: np.array(
                samples,
                dtype=np.float32
            )
            for name, samples in data.items()
        }

    except Exception as e:

        print(
            f"Erreur lecture base : {e}"
        )

        return {}


def save_db(db):

    DB_DIR.mkdir(
        exist_ok=True
    )

    DB_FILE.write_text(
        json.dumps(
            {
                name: samples.tolist()
                for name, samples in db.items()
            },
            indent=2
        ),
        encoding="utf-8"
    )


def cosine(a, b):

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
            + 1e-9
        )
    )


def identify(embedding, db, threshold):

    best_name = None
    best_score = -1.0

    for name, samples in db.items():

        score = max(
            cosine(
                embedding,
                sample
            )
            for sample in samples
        )

        if score > best_score:

            best_name = name
            best_score = score

    if best_score >= threshold:

        return (
            best_name,
            best_score
        )

    return (
        None,
        best_score
    )


# ============================================================
# CAMERA
# ============================================================

def open_cam(source):

    if str(source).isdigit():

        backend = (
            cv2.CAP_DSHOW
            if sys.platform.startswith("win")
            else cv2.CAP_ANY
        )

        cap = cv2.VideoCapture(
            int(source),
            backend
        )

    else:

        cap = cv2.VideoCapture(
            source
        )

    if not cap.isOpened():

        sys.exit(
            f"Impossible d'ouvrir la caméra : {source}"
        )

    return cap


# ============================================================
# AFFICHAGE
# ============================================================

def draw_face(
    frame,
    face,
    label,
    color
):

    x, y, w, h = map(
        int,
        face[:4]
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


# ============================================================
# LIVENESS / ANTI-PHOTO
# ============================================================

class LivenessDetector:

    def __init__(self):
        self.reset()

    def reset(self):

        self.challenge = random.choice(
            [
                "GAUCHE",
                "DROITE"
            ]
        )

        self.started = False
        self.start_x = None

        self.progress = 0

        self.success = False
        self.failed = False

        self.start_time = None

    # --------------------------------------------------------
    # DEMARRAGE DU TEST
    # --------------------------------------------------------

    def start(self):

        if self.start_time is None:

            self.start_time = time.monotonic()

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self, face):

        # Si le test est déjà terminé,
        # on ne fait plus rien.
        if self.success or self.failed:
            return self.success

        # Aucun visage
        if face is None:

            # Le chronomètre continue seulement
            # s'il avait déjà commencé.
            if self.start_time is not None:

                if (
                    time.monotonic()
                    - self.start_time
                    >= LIVENESS_TIMEOUT
                ):

                    self.failed = True

            return False

        # ----------------------------------------------------
        # LANDMARKS
        # ----------------------------------------------------

        landmarks = face[4:14]

        if len(landmarks) < 6:
            return False

        x, y, w, h = face[:4]

        # YuNet :
        #
        # 0 = oeil droit X
        # 1 = oeil droit Y
        # 2 = oeil gauche X
        # 3 = oeil gauche Y
        # 4 = nez X
        # 5 = nez Y
        #
        nose_x = float(
            landmarks[4]
        )

        # Position du nez normalisée
        normalized_x = (
            nose_x - x
        ) / max(w, 1)

        # ----------------------------------------------------
        # DEMARRAGE DU CHRONOMETRE
        # ----------------------------------------------------

        self.start()

        # ----------------------------------------------------
        # PREMIERE POSITION
        # ----------------------------------------------------

        if not self.started:

            self.start_x = normalized_x
            self.started = True

            return False

        # ----------------------------------------------------
        # VERIFICATION DU TIMEOUT
        # ----------------------------------------------------

        elapsed = (
            time.monotonic()
            - self.start_time
        )

        if elapsed >= LIVENESS_TIMEOUT:

            self.failed = True

            return False

        # ----------------------------------------------------
        # MOUVEMENT
        # ----------------------------------------------------

        movement = (
            normalized_x
            - self.start_x
        )

        # Seuil du mouvement
        threshold = 0.10

        movement_detected = False

        if self.challenge == "GAUCHE":

            if movement < -threshold:

                movement_detected = True

        else:

            if movement > threshold:

                movement_detected = True

        # ----------------------------------------------------
        # CONFIRMATION DU MOUVEMENT
        # ----------------------------------------------------

        if movement_detected:

            self.progress += 1

        else:

            # Les frames précédentes ne comptent plus
            # si le mouvement n'est pas maintenu.
            self.progress = 0

        # ----------------------------------------------------
        # LIVENESS OK
        # ----------------------------------------------------

        if (
            self.progress
            >= LIVENESS_CONFIRM_FRAMES
        ):

            self.success = True

            return True

        return False

    # --------------------------------------------------------
    # TEMPS RESTANT
    # --------------------------------------------------------

    def remaining_time(self):

        if self.start_time is None:
            return LIVENESS_TIMEOUT

        if self.success:
            return 0.0

        if self.failed:
            return 0.0

        elapsed = (
            time.monotonic()
            - self.start_time
        )

        remaining = (
            LIVENESS_TIMEOUT
            - elapsed
        )

        return max(
            0.0,
            remaining
        )

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    def message(self):

        if self.success:
            return "LIVENESS OK"

        if self.failed:
            return "!!! INTRUS / SPOOFING !!!"

        if self.challenge == "GAUCHE":
            return "TOURNEZ LA TETE A GAUCHE"

        return "TOURNEZ LA TETE A DROITE"


# ============================================================
# ENROLEMENT
# ============================================================

def cmd_enroll(args):

    engine = FaceEngine()

    db = load_db()

    cap = open_cam(
        args.source
    )

    samples = []

    last_capture = 0

    print()
    print("=" * 60)
    print("ENROLEMENT SECURISE")
    print("=" * 60)
    print()

    print(
        f"Personne : {args.name}"
    )

    print()

    print(
        "Regardez la caméra."
    )

    print(
        "Bougez légèrement la tête."
    )

    print()

    while len(samples) < args.samples:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.resize(
            frame,
            (W, H)
        )

        faces = engine.detect(
            frame
        )

        if len(faces) == 1:

            face = faces[0]

            if (
                time.time()
                - last_capture
                > 0.4
            ):

                embedding = engine.embed(
                    frame,
                    face
                )

                samples.append(
                    embedding
                )

                last_capture = time.time()

            label = (
                f"Capture "
                f"{len(samples)}/"
                f"{args.samples}"
            )

            color = (
                0,
                255,
                0
            )

        elif len(faces) == 0:

            label = "AUCUN VISAGE"

            color = (
                0,
                0,
                255
            )

        else:

            label = "UN SEUL VISAGE"

            color = (
                0,
                0,
                255
            )

        for face in faces:

            draw_face(
                frame,
                face,
                label,
                color
            )

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow(
            "Enrolement",
            frame
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break

    cap.release()

    cv2.destroyAllWindows()

    if len(samples) < args.samples:

        print(
            "Enrolement annulé."
        )

        return

    db[args.name] = np.array(
        samples,
        dtype=np.float32
    )

    save_db(db)

    print()

    print(
        f"OK : {args.name} enregistré."
    )


# ============================================================
# AFFICHAGE ALERTE SPOOFING
# ============================================================

def show_spoofing_alert(cap):

    alert_start = time.monotonic()

    while (
        time.monotonic()
        - alert_start
        < ALERT_DISPLAY_TIME
    ):

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.resize(
            frame,
            (W, H)
        )

        # Bordure rouge
        cv2.rectangle(
            frame,
            (0, 0),
            (W - 1, H - 1),
            (0, 0, 255),
            10
        )

        # Alerte principale
        cv2.putText(
            frame,
            "!!! INTRUSION !!!",
            (70, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            4
        )

        cv2.putText(
            frame,
            "SPOOFING DETECTE",
            (100, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "TEST DE VIVACITE ECHOUE",
            (75, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ASPHALEIA - Anti-Spoofing",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            return False

    return True


# ============================================================
# SURVEILLANCE
# ============================================================

def send_face_status(status_url, frame, boxes, liveness_state, message, intrusion):
    """Envoie au dashboard l'image webcam + état de la vision."""
    if not status_url:
        return False

    try:
        ok_jpg, jpg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        if not ok_jpg:
            return False

        payload = {
            "w": int(frame.shape[1]),
            "h": int(frame.shape[0]),
            "faces": boxes,
            "liveness": liveness_state,
            "message": message,
            "intrusion": bool(intrusion),
            "frame": base64.b64encode(jpg.tobytes()).decode("ascii")
        }

        sent, _ = send_alert(status_url, payload, timeout=0.5)
        return sent
    except Exception:
        return False


def cmd_run(args):
    db = load_db()

    if not db:
        sys.exit(
            "Aucune personne enregistrée.\n"
            "Utilise : python3 face.py enroll Melissa"
        )

    engine = FaceEngine()

    OUT.mkdir(exist_ok=True)
    unknown_dir = OUT / "inconnus"
    unknown_dir.mkdir(exist_ok=True)

    cap = open_cam(args.source)

    # --------------------------------------------------------
    # CONNEXION DASHBOARD
    # --------------------------------------------------------
    status_url = None

    if args.api:
        api_base = args.api.rstrip("/")
        status_url = api_base + "/api/face/status"

        if not args.alert_url:
            args.alert_url = api_base + "/api/alerts"

        print(f"[ASPHALEIA] API dashboard : {api_base}")

    # --------------------------------------------------------
    # ETAT
    # --------------------------------------------------------
    liveness = LivenessDetector()
    liveness_ok = False
    liveness_owner = None

    intrusion = False
    spoof_until = 0.0
    last_alert = 0.0
    last_push = 0.0
    last_api_fail = 0.0
    consecutive_unknown = 0

    print()
    print("=" * 60)
    print("ASPHALEIA")
    print("=" * 60)
    print("Personnes autorisées :")
    for name in db:
        print(f"  - {name}")
    print()
    print("q = quitter")
    print()

    try:
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()

            if not ok:
                print("Impossible de lire la caméra.")
                break

            frame = cv2.resize(frame, (W, H))
            now = time.monotonic()

            faces = engine.detect(frame)

            # =================================================
            # RECONNAISSANCE : TOUJOURS AVANT LA LIVENESS
            # =================================================
            known_faces = []
            unknown_faces = []

            for face in faces:
                try:
                    embedding = engine.embed(frame, face)
                    name, score = identify(
                        embedding,
                        db,
                        args.threshold
                    )
                except Exception:
                    name, score = None, -1.0

                if name:
                    known_faces.append((face, name, score))
                else:
                    unknown_faces.append((face, score))

            # =================================================
            # 1. PERSONNE INCONNUE = INTRUS IMMEDIATEMENT
            # =================================================
            if unknown_faces:
                consecutive_unknown += 1
                intrusion = True
                liveness_ok = False
                liveness_owner = None
            else:
                consecutive_unknown = 0

            # =================================================
            # 2. PERSONNE CONNUE = TEST DE VIVACITE
            # =================================================
            if not unknown_faces and len(known_faces) == 1:
                face, name, score = known_faces[0]

                if liveness_owner != name:
                    liveness.reset()
                    liveness_owner = name
                    liveness_ok = False
                    intrusion = False

                if not liveness_ok:
                    liveness_ok = liveness.update(face)

                if liveness.failed:
                    intrusion = True

            elif not unknown_faces and len(known_faces) == 0:
                # Pas de visage : on garde le challenge s'il était déjà commencé.
                if liveness.start_time is not None and not liveness_ok:
                    if now - liveness.start_time >= LIVENESS_TIMEOUT:
                        liveness.failed = True
                        intrusion = True

            elif len(faces) > 1:
                # Plusieurs visages : accès non autorisé tant qu'on ne sait
                # pas identifier une seule personne et valider sa vivacité.
                liveness_ok = False
                intrusion = True

            # =================================================
            # 3. PHOTO D'UNE PERSONNE CONNUE = SPOOFING
            # =================================================
            if liveness.failed:
                intrusion = True
                spoof_until = max(
                    spoof_until,
                    time.monotonic() + ALERT_DISPLAY_TIME
                )

                if (
                    args.alert_url
                    and time.time() - last_alert >= args.cooldown
                ):
                    payload = {
                        "source": "vision_face",
                        "type": "spoofing",
                        "severity": "high",
                        "reason": "liveness_timeout",
                        "message": "Le mouvement demandé n'a pas été détecté.",
                        "challenge": liveness.challenge,
                        "timestamp": datetime.now().isoformat(timespec="seconds")
                    }

                    sent, info = send_alert(args.alert_url, payload)
                    last_alert = time.time()

                    if sent:
                        print("[DASHBOARD] Alerte SPOOFING envoyée.")
                    else:
                        print(f"[DASHBOARD] Erreur API : {info}")

                # On attend la fin de l'affichage avant de recommencer
                # un nouveau challenge.
                if time.monotonic() >= spoof_until:
                    liveness.reset()
                    liveness_owner = None
                    liveness_ok = False
                    intrusion = False
                    consecutive_unknown = 0
                    spoof_until = 0.0

            # =================================================
            # 4. ALERTE INTRUSION : PERSONNE INCONNUE
            # =================================================
            if (
                unknown_faces
                and args.alert_url
                and time.time() - last_alert >= args.cooldown
            ):
                filename = f"intrus_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                snapshot = unknown_dir / filename

                alert_frame = frame.copy()
                cv2.putText(
                    alert_frame,
                    "INTRUS",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )
                cv2.imwrite(str(snapshot), alert_frame)

                print()
                print("[ALERTE] PERSONNE INCONNUE")
                print(f"[ALERTE] Capture : {snapshot}")

                payload = {
                    "source": "vision_face",
                    "type": "unknown_face",
                    "severity": "high",
                    "liveness": False,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "snapshot": filename
                }

                sent, info = send_alert(args.alert_url, payload)
                last_alert = time.time()

                if sent:
                    print("[DASHBOARD] Alerte INTRUS envoyée.")
                else:
                    print(f"[DASHBOARD] Erreur API : {info}")

            # =================================================
            # MESSAGE POUR LE DASHBOARD
            # =================================================
            if unknown_faces:
                live_state = "failed"
                live_msg = "!!! INTRUS / ACCÈS NON AUTORISÉ !!!"
            elif liveness.failed:
                live_state = "failed"
                live_msg = "!!! INTRUS / SPOOFING !!!"
            elif liveness_ok:
                live_state = "ok"
                live_msg = "LIVENESS OK"
            elif len(known_faces) == 1:
                live_state = "pending"
                live_msg = (
                    f"{liveness.message()} "
                    f"[{liveness.remaining_time():.1f} s]"
                )
            elif len(faces) > 1:
                live_state = "failed"
                live_msg = "PLUSIEURS VISAGES - ACCÈS REFUSÉ"
            else:
                live_state = "waiting"
                live_msg = "PLACEZ-VOUS DEVANT LA CAMERA"

            # =================================================
            # AFFICHAGE LOCAL
            # =================================================
            boxes = []

            for face, name, score in known_faces:
                boxes.append({
                    "box": [int(v) for v in face[:4]],
                    "kind": "known",
                    "label": f"{name} {score:.2f}",
                    "score": float(score)
                })

            for face, score in unknown_faces:
                boxes.append({
                    "box": [int(v) for v in face[:4]],
                    "kind": "unknown",
                    "label": f"INTRUS {score:.2f}",
                    "score": float(score)
                })

            if not unknown_faces and not liveness_ok:
                for face in faces:
                    if not any(face is item[0] for item in known_faces):
                        boxes.append({
                            "box": [int(v) for v in face[:4]],
                            "kind": "pending",
                            "label": "VERIFICATION",
                            "score": 0.0
                        })

            if not args.no_display:
                if unknown_faces:
                    message = "!!! INTRUS / ACCES NON AUTORISE !!!"
                    color = (0, 0, 255)
                elif liveness.failed:
                    message = "!!! INTRUS / SPOOFING !!!"
                    color = (0, 0, 255)
                elif liveness_ok:
                    message = "LIVENESS OK - ACCES AUTORISE"
                    color = (0, 255, 0)
                elif len(known_faces) == 1:
                    message = (
                        f"{liveness.message()} "
                        f"[{liveness.remaining_time():.1f} s]"
                    )
                    color = (0, 255, 255)
                elif len(faces) > 1:
                    message = "PLUSIEURS VISAGES - ACCES REFUSE"
                    color = (0, 0, 255)
                else:
                    message = "PLACEZ-VOUS DEVANT LA CAMERA"
                    color = (0, 255, 255)

                cv2.putText(
                    frame,
                    message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

                for face, name, score in known_faces:
                    draw_face(
                        frame,
                        face,
                        f"{name} {score:.2f}",
                        (0, 255, 0) if liveness_ok else (0, 255, 255)
                    )

                for face, score in unknown_faces:
                    draw_face(
                        frame,
                        face,
                        f"INTRUS {score:.2f}",
                        (0, 0, 255)
                    )

                if intrusion:
                    cv2.rectangle(
                        frame,
                        (0, 0),
                        (W - 1, H - 1),
                        (0, 0, 255),
                        7
                    )
                    cv2.putText(
                        frame,
                        "!!! INTRUSION !!!",
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

                if liveness.failed and time.monotonic() < spoof_until:
                    cv2.putText(
                        frame,
                        "SPOOFING DETECTE",
                        (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        3
                    )

                elapsed_ms = (time.perf_counter() - start) * 1000
                cv2.putText(
                    frame,
                    f"{elapsed_ms:.1f} ms",
                    (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # ENVOI DU VRAI FLUX AU DASHBOARD
            # IMPORTANT : on envoie APRES les overlays.
            # =================================================
            if (
                status_url
                and time.monotonic() - last_push >= 0.20
            ):
                last_push = time.monotonic()
                sent = send_face_status(
                    status_url,
                    frame,
                    boxes,
                    live_state,
                    live_msg,
                    intrusion
                )

                if not sent and time.monotonic() - last_api_fail > 3:
                    last_api_fail = time.monotonic()

            # =================================================
            # FENETRE LOCALE
            # =================================================
            if not args.no_display:
                cv2.imshow("ASPHALEIA - Anti-Spoofing", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nArrêt...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# API
# ============================================================

def send_alert(
    url,
    payload,
    timeout=2
):

    request = urllib.request.Request(
        url,
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:

            return (
                True,
                response.status
            )

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError
    ) as e:

        return (
            False,
            str(e)
        )


# ============================================================
# LIST
# ============================================================

def cmd_list(args):

    db = load_db()

    if not db:

        print(
            "Aucune personne enregistrée."
        )

        return

    print(
        "Personnes enregistrées :"
    )

    for name, samples in db.items():

        print(
            f"  - {name} "
            f"({len(samples)} empreintes)"
        )


# ============================================================
# DELETE
# ============================================================

def cmd_delete(args):

    db = load_db()

    if args.name not in db:

        print(
            f"{args.name} n'existe pas."
        )

        return

    del db[args.name]

    save_db(db)

    print(
        f"{args.name} supprimé."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "ASPHALEIA - "
        "Reconnaissance faciale "
        "+ anti-spoofing"
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True
    )

    # ========================================================
    # ENROLL
    # ========================================================

    enroll = sub.add_parser(
        "enroll"
    )

    enroll.add_argument(
        "name"
    )

    enroll.add_argument(
        "--samples",
        type=int,
        default=20
    )

    enroll.add_argument(
        "--source",
        default="0"
    )

    enroll.set_defaults(
        fn=cmd_enroll
    )

    # ========================================================
    # RUN
    # ========================================================

    run = sub.add_parser(
        "run"
    )

    run.add_argument(
        "--source",
        default="0"
    )

    run.add_argument(
        "--threshold",
        type=float,
        default=COSINE_THRESHOLD
    )

    run.add_argument(
        "--confirm-frames",
        type=int,
        default=CONFIRM_FRAMES
    )

    run.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN
    )

    run.add_argument(
        "--alert-url",
        default=None
    )

    run.add_argument(
        "--api",
        default=None,
        help="URL de l'API ASPHALEIA, ex : http://localhost:8000"
    )

    run.add_argument(
        "--no-display",
        action="store_true"
    )

    run.set_defaults(
        fn=cmd_run
    )

    # ========================================================
    # LIST
    # ========================================================

    list_cmd = sub.add_parser(
        "list"
    )

    list_cmd.set_defaults(
        fn=cmd_list
    )

    # ========================================================
    # DELETE
    # ========================================================

    delete = sub.add_parser(
        "delete"
    )

    delete.add_argument(
        "name"
    )

    delete.set_defaults(
        fn=cmd_delete
    )

    # ========================================================
    # EXECUTION
    # ========================================================

    args = parser.parse_args()

    args.fn(args)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()