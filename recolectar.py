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

# Cada muestra equivale a 30 frames (aprox. 1 segundo de video)
MUESTRAS_POR_TANDA = 10 

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

    # Abrimos el archivo en modo 'append' para no borrar datos anteriores
    with open(RUTA_DATOS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Se perdio la senal de la camara.")
                break

            frame = cv2.flip(frame, 1) # Efecto espejo
            alto, ancho = frame.shape[:2] 
            lm, lado = detector.procesar(frame)

            feats_frame = None
            if lm is not None:
                detector.dibujar(frame, lm)
                feats_frame = normalizar_landmarks(lm, lado, ancho, alto)

            # Lógica de grabación
            if grabando and etiqueta and feats_frame is not None:
                buffer.append(feats_frame)
                # Cuando el buffer se llena (30 frames), se guarda una muestra
                if len(buffer) == VENTANA_TEMPORAL:
                    feats = features_ventana(np.array(buffer))
                    if feats is not None:
                        writer.writerow([etiqueta, *feats.round(5).tolist()])
                        conteo[etiqueta] += 1
                        muestras_tanda += 1
                        f.flush() # Fuerza la escritura en disco
                        buffer.clear()
                        if muestras_tanda >= MUESTRAS_POR_TANDA:
                            grabando = False # Se detiene la grabación tras 10 muestras
            elif not grabando:
                buffer.clear()

            # ---- Interfaz Gráfica de la cámara ----
            estado = "GRABANDO" if grabando else "PAUSADO"
            color = (0, 0, 255) if grabando else (0, 255, 255) # Rojo si graba, amarillo si pausa
            
            texto(frame, f"Letra: {etiqueta or '-'}  |  {estado}  |  buf {len(buffer)}/{VENTANA_TEMPORAL}", 30, color)
            texto(frame, "Mano: " + ("detectada" if lm is not None else "NO detectada"), 60,
                  (0, 255, 0) if lm is not None else (0, 0, 255))
            texto(frame, f"Muestras tanda: {muestras_tanda}/{MUESTRAS_POR_TANDA}", 90, escala=0.6)
            
            # Resumen de conteo
            resumen = "  ".join(f"{k}:{v}" for k, v in sorted(conteo.items()))
            texto(frame, resumen[:400], alto - 60, escala=0.5, color=(255, 255, 0)) 
            
            texto(frame, "A-Z | elegir | ESPACIO grabar | ESC salir", alto - 30, escala=0.7, color=(255, 255, 200)) 

            cv2.imshow("Recolector de senas", frame)
            tecla = cv2.waitKey(1) & 0xFF

            # ---- Control de teclado ----
            if tecla == 27: # ESC para salir
                break
            elif tecla == 32: # ESPACIO para grabar/pausar
                if etiqueta:
                    grabando = not grabando
                    if grabando:
                        muestras_tanda = 0
                        buffer.clear()
            # Se aceptan letras de la A a la Z y la letra Ñ
            elif (ord("a") <= tecla <= ord("z")) or tecla == 241: # 241 es el código ASCII de la ñ minúscula
                etiqueta = chr(tecla).upper()
                grabando = False
                muestras_tanda = 0
                buffer.clear()

    cap.release()
    detector.cerrar()
    cv2.destroyAllWindows()
    print("Muestras totales por letra:", dict(sorted(conteo.items())))

if __name__ == "__main__":
    main()