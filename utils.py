"""Utilidades compartidas: cámara, detector de manos, normalización y features temporales."""
import os
import platform
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

CARPETA_DATOS = "datos"
RUTA_DATOS = os.path.join(CARPETA_DATOS, "muestras.csv")
RUTA_MODELO = os.path.join(CARPETA_DATOS, "modelo.joblib")

NUM_PUNTOS = 21
FEATURES_POR_FRAME = NUM_PUNTOS * 3  # 63

VENTANA_TEMPORAL = 30  # frames por muestra (~1s a 30fps)

# Features temporales por ventana:
#   media (63) + desplazamiento (63) + varianza (63) + velocidad (63) + recorrido (21)
NUM_CARACTERISTICAS = FEATURES_POR_FRAME * 4 + NUM_PUNTOS  # 273


def abrir_camara(indice=0, ancho=1280, alto=720):
    """Abre la cámara. En Windows usa DirectShow porque arranca mucho más rápido."""
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(indice)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    return cap


def normalizar_landmarks(hand_landmarks, lado_mano, ancho, alto):
    """
    Convierte 21 puntos en un vector 63 invariante a:
      - posición en pantalla (relativo a la muñeca)
      - distancia a la cámara (escala por tamaño de mano)
      - mano usada (izquierda se refleja)
    """
    pts = np.array(
        [[lm.x * ancho, lm.y * alto, lm.z * ancho] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    pts -= pts[0]  # muñeca como origen

    if lado_mano == "Left":
        pts[:, 0] *= -1  # espejo horizontal

    escala = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if escala < 1e-6:
        return None
    pts /= escala
    return pts.flatten()  # (63,)


def features_ventana(ventana):
    """
    ventana: array (N, 63) de frames normalizados.
    Devuelve vector de 273 features que resume el movimiento.
    """
    v = np.asarray(ventana, dtype=np.float32)
    n = len(v)
    if n < 2:
        return None

    media = v.mean(axis=0)                       # 63
    desplazamiento = v[-1] - v[0]                # 63
    varianza = v.var(axis=0)                     # 63
    velocidad = np.abs(np.diff(v, axis=0)).mean(axis=0)  # 63

    # recorrido por punto (21)
    tray = np.diff(v.reshape(n, NUM_PUNTOS, 3), axis=0)  # (n-1, 21, 3)
    recorrido = np.linalg.norm(tray, axis=2).sum(axis=0)  # (21,)

    return np.concatenate([media, desplazamiento, varianza, velocidad, recorrido])


class DetectorManos:
    """Envoltorio de MediaPipe Hands."""

    def __init__(self, max_manos=1, complejidad=1, conf_deteccion=0.7, conf_seguimiento=0.5):
        self.mp_hands = mp.solutions.hands
        self.dibujo = mp.solutions.drawing_utils
        self.estilos = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_manos,
            model_complexity=complejidad,
            min_detection_confidence=conf_deteccion,
            min_tracking_confidence=conf_seguimiento,
        )

    def procesar(self, frame_bgr):
        """Devuelve (landmarks, 'Left'/'Right') de la primera mano, o (None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lado = res.multi_handedness[0].classification[0].label
        return res.multi_hand_landmarks[0], lado

    def dibujar(self, frame_bgr, hand_landmarks):
        self.dibujo.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.estilos.get_default_hand_landmarks_style(),
            self.estilos.get_default_hand_connections_style(),
        )

    def cerrar(self):
        self.hands.close()