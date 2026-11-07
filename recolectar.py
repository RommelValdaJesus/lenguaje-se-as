import csv
import os
from collections import Counter, deque

import cv2
import numpy as np

from utils import (
    CARPETA_DATOS,
    NUM_CARACTERISTICAS,
    RUTA_DATOS,
    VENTANA_TEMPORAL,
    DetectorManos,
    abrir_camara,
    features_ventana,
    normalizar_landmarks,
)

MUESTRAS_POR_TANDA = 10  # cada muestra son 30 frames; 10 ≈ 300 frames ≈ 10s


def contar_existentes():
    conteo = Counter()
    if os.path.exists(RUTA_DATOS):
        with open(RUTA_DATOS, newline="", encoding="utf-8") as f:
            for fila in csv.reader(f):
                if len(fila) == NUM_CARACTERISTICAS + 1:
                    conteo[fila[0]] += 1
    return conteo


def texto(frame, msg, y, color=(255, 255, 255), escala=0.7):
    cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


def main():
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    cap = abrir_camara()
    if not cap.isOpened():
        print("No se pudo abrir la camara. Revisa que no la use otra app.")
        return

    detector = DetectorManos()
    conteo = contar_existentes()
    etiqueta = None
    grabando = False
    muestras_tanda = 0
    buffer = deque(maxlen=VENTANA_TEMPORAL)

    with open(RUTA_DATOS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Se perdio la senal de la camara.")
                break

            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            lm, lado = detector.procesar(frame)

            feats_frame = None
            if lm is not None:
                detector.dibujar(frame, lm)
                feats_frame = normalizar_landmarks(lm, lado, ancho, alto)

            # Acumular solo si estamos grabando y hay mano
            if grabando and etiqueta and feats_frame is not None:
                buffer.append(feats_frame)
                if len(buffer) == VENTANA_TEMPORAL:
                    feats = features_ventana(np.array(buffer))
                    if feats is not None:
                        writer.writerow([etiqueta, *feats.round(5).tolist()])
                        conteo[etiqueta] += 1
                        muestras_tanda += 1
                        f.flush()
                        buffer.clear()
                        if muestras_tanda >= MUESTRAS_POR_TANDA:
                            grabando = False
            elif not grabando:
                buffer.clear()

            estado = "GRABANDO" if grabando else "PAUSADO"
            color = (0, 0, 255) if grabando else (0, 255, 255)
            texto(frame, f"Sena: {etiqueta or '-'}  |  {estado}  |  buf {len(buffer)}/{VENTANA_TEMPORAL}", 30, color)
            texto(frame, "Mano: " + ("detectada" if lm is not None else "NO detectada"), 60,
                  (0, 255, 0) if lm is not None else (0, 0, 255))
            texto(frame, f"Muestras tanda: {muestras_tanda}/{MUESTRAS_POR_TANDA}", 90, escala=0.6)
            resumen = "  ".join(f"{k}:{v}" for k, v in sorted(conteo.items()))
            texto(frame, resumen[:400], alto - 60, escala=0.5, color=(255, 255, 0)) 
            texto(frame, "A-Z/0-9 elegir | ESPACIO grabar | ESC salir", alto - 30, escala=0.7, color=(255, 255, 200)) 

            cv2.imshow("Recolector de senas", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                break
            elif tecla == 32:
                if etiqueta:
                    grabando = not grabando
                    if grabando:
                        muestras_tanda = 0
                        buffer.clear()
            elif ord("a") <= tecla <= ord("z") or ord("0") <= tecla <= ord("9"):
                etiqueta = chr(tecla).upper()
                grabando = False
                muestras_tanda = 0
                buffer.clear()

    cap.release()
    detector.cerrar()
    cv2.destroyAllWindows()
    print("Muestras totales por seña:", dict(sorted(conteo.items())))


if __name__ == "__main__":
    main()