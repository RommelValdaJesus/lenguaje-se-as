import os
import tkinter as tk
from collections import Counter, deque
from tkinter import messagebox, ttk

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from utils import (
    RUTA_MODELO,
    VENTANA_TEMPORAL,
    DetectorManos,
    abrir_camara,
    features_ventana,
    normalizar_landmarks,
)

UMBRAL_CONFIANZA = 0.60   
VENTANA_SUAVIZADO = 3     
PREDICCIONES_ESTABLES = 3 
ANCHO_VIDEO = 640


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor de Lengua de Señas a Texto - TECBA")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("900x500") 

        # Verificar modelo
        if not os.path.exists(RUTA_MODELO):
            messagebox.showerror("Sin modelo", "No existe datos/modelo.joblib.\nEjecuta recolectar.py y entrenar.py primero.")
            self.root.destroy()
            return

        # Cargar modelo
        try:
            datos = joblib.load(RUTA_MODELO)
            self.modelo = datos["modelo"]
            self.modelo.n_jobs = 1
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
            self.root.destroy()
            return

        # Abrir cámara
        self.cap = abrir_camara()
        if not self.cap.isOpened():
            messagebox.showerror("Cámara", "No se pudo abrir la cámara. Verifica que no esté en uso.")
            self.root.destroy()
            return

        self.detector = DetectorManos()
        self.buffer = deque(maxlen=VENTANA_TEMPORAL)
        self.historial = deque(maxlen=VENTANA_SUAVIZADO)
        self.ultima = None
        self.contador = 0
        self.bloqueada = False
        self.texto = ""
        self.job = None
        self._imgtk = None

        self._construir_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)
        
        # Atajos de teclado
        self.root.bind("<space>", lambda e: self.agregar(" "))
        self.root.bind("<BackSpace>", lambda e: self.borrar())
        
        self.actualizar()

    # ---------- INTERFAZ GRÁFICA ----------
    def _construir_ui(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")

        cont = tk.Frame(self.root, bg="#1e1e1e")
        cont.pack(padx=10, pady=10)

        # Panel Izquierdo: Video
        self.lbl_video = tk.Label(cont, bg="black")
        self.lbl_video.grid(row=0, column=0, rowspan=7, padx=(0, 10))

        # Panel Derecho: Predicción y Confianza
        tk.Label(cont, text="Seña detectada", fg="#aaa", bg="#1e1e1e", font=("Segoe UI", 11)).grid(row=0, column=1, sticky="n")
        self.lbl_pred = tk.Label(cont, text="-", fg="#4ade80", bg="#1e1e1e", font=("Segoe UI", 72, "bold"), width=3)
        self.lbl_pred.grid(row=1, column=1)

        self.lbl_conf = tk.Label(cont, text="Confianza: 0%", fg="#ddd", bg="#1e1e1e", font=("Segoe UI", 11))
        self.lbl_conf.grid(row=2, column=1)

        self.lbl_buf = tk.Label(cont, text=f"Buffer: 0/{VENTANA_TEMPORAL}", fg="#aaa", bg="#1e1e1e", font=("Segoe UI", 10))
        self.lbl_buf.grid(row=3, column=1)

        self.progreso = ttk.Progressbar(cont, length=200, maximum=PREDICCIONES_ESTABLES)
        self.progreso.grid(row=4, column=1, pady=5)

        # Botones de Control
        botones = tk.Frame(cont, bg="#1e1e1e")
        botones.grid(row=5, column=1, pady=10)
        
        ttk.Button(botones, text="Espacio", command=lambda: self.agregar(" ")).pack(fill="x", pady=2)
        ttk.Button(botones, text="Borrar", command=self.borrar).pack(fill="x", pady=2)
        ttk.Button(botones, text="Limpiar", command=self.limpiar).pack(fill="x", pady=2)

        # Cuadro de texto final
        self.txt = tk.Text(self.root, height=3, font=("Segoe UI", 16), wrap="word", bg="#2a2a2a", fg="white", insertbackground="white")
        self.txt.pack(fill="x", padx=10, pady=(0, 10))

    # ---------- LÓGICA DE VISIÓN Y PREDICCIÓN ----------
    def actualizar(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1) # Espejo
            alto, ancho = frame.shape[:2]
            lm, lado = self.detector.procesar(frame)

            feats_frame = None
            if lm is not None:
                self.detector.dibujar(frame, lm)
                feats_frame = normalizar_landmarks(lm, lado, ancho, alto)

            pred, conf = None, 0.0
            if feats_frame is not None:
                self.buffer.append(feats_frame)
            else:
                self.buffer.clear()

            # Predicción cuando el buffer está lleno
            if len(self.buffer) == VENTANA_TEMPORAL:
                feats = features_ventana(np.array(self.buffer))
                if feats is not None:
                    try:
                        proba = self.modelo.predict_proba(feats.reshape(1, -1))[0]
                        i = int(np.argmax(proba))
                        conf = float(proba[i])
                        if conf >= UMBRAL_CONFIANZA:
                            pred = self.modelo.classes_[i]
                    except Exception:
                        pass 
                    
                    # Ventana deslizante
                    self.buffer.popleft()

            self.historial.append(pred)
            estable = self._suavizar()
            self._procesar_estabilidad(estable)

            # Actualizar UI
            self.lbl_pred.config(text=estable or "-")
            self.lbl_conf.config(text=f"Confianza: {conf * 100:.0f}%")
            self.lbl_buf.config(text=f"Buffer: {len(self.buffer)}/{VENTANA_TEMPORAL}")
            self.progreso["value"] = 0 if self.bloqueada else self.contador
            self._mostrar(frame)

        self.job = self.root.after(15, self.actualizar)

    def _suavizar(self):
        
        validos = [p for p in self.historial if p is not None]
        if not validos:
            return None
        seña, votos = Counter(validos).most_common(1)[0]
        return seña if votos > len(self.historial) // 2 else None

    def _procesar_estabilidad(self, seña):
        
        if seña is None:
            self.ultima, self.contador, self.bloqueada = None, 0, False
            return
        if seña != self.ultima:
            self.ultima, self.contador, self.bloqueada = seña, 1, False
            return
        if self.bloqueada:
            return
        
        self.contador += 1
        if self.contador >= PREDICCIONES_ESTABLES:
            self.agregar(seña)
            self.bloqueada = True # Bajar la mano para volver a escribir

    def _mostrar(self, frame):
        alto, ancho = frame.shape[:2]
        nuevo_alto = int(alto * ANCHO_VIDEO / ancho)
        frame = cv2.resize(frame, (ANCHO_VIDEO, nuevo_alto))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._imgtk = ImageTk.PhotoImage(image=img)
        self.lbl_video.config(image=self._imgtk)

    # ---------- CONTROL DE TEXTO ----------
    def agregar(self, c):
        self.texto += c
        self._refrescar_texto()

    def borrar(self):
        self.texto = self.texto[:-1]
        self._refrescar_texto()

    def limpiar(self):
        self.texto = ""
        self._refrescar_texto()

    def _refrescar_texto(self):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", self.texto)

    def cerrar(self):
        if self.job:
            self.root.after_cancel(self.job)
        self.cap.release()
        self.detector.cerrar()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    try:
        root.mainloop()
    except tk.TclError:
        pass