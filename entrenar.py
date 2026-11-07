import csv
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import NUM_CARACTERISTICAS, RUTA_DATOS, RUTA_MODELO

MINIMO_POR_SENA = 5  # cada muestra = 30 frames; 30 muestras ≈ 900 frames


def cargar():
    X, y = [], []
    with open(RUTA_DATOS, newline="", encoding="utf-8") as f:
        for fila in csv.reader(f):
            if len(fila) != NUM_CARACTERISTICAS + 1:
                continue
            y.append(fila[0])
            X.append([float(v) for v in fila[1:]])
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    if not os.path.exists(RUTA_DATOS):
        sys.exit("No existe datos/muestras.csv. Ejecuta primero recolectar.py")

    X, y = cargar()
    conteo = Counter(y)
    print("Muestras por seña:", dict(sorted(conteo.items())))

    if len(conteo) < 2:
        sys.exit("Necesitas al menos 2 señas distintas para entrenar.")
    pocas = [k for k, v in conteo.items() if v < MINIMO_POR_SENA]
    if pocas:
        sys.exit(f"Estas señas tienen menos de {MINIMO_POR_SENA} muestras: {pocas}. Graba mas.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modelo = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    modelo.fit(X_tr, y_tr)

    pred = modelo.predict(X_te)
    print("\n=== Reporte ===")
    print(classification_report(y_te, pred, digits=3))
    print("Clases:", list(modelo.classes_))
    print(confusion_matrix(y_te, pred, labels=modelo.classes_))

    modelo.fit(X, y)
    modelo.n_jobs = 1
    joblib.dump({"modelo": modelo, "clases": list(modelo.classes_)}, RUTA_MODELO)
    print(f"\nModelo guardado en {RUTA_MODELO}")
    print("Recuerda probar en vivo con otra persona y a distintas velocidades.")


if __name__ == "__main__":
    main()