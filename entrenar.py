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

# Cada muestra = 30 frames. Con 10 muestras (300 frames) ya se puede entrenar.
MINIMO_POR_SENA = 10 

# Lista de letras permitidas (A-Z + Ñ)
LETRAS_VALIDAS = set("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")


def cargar():
    
    X, y = [], []
    if not os.path.exists(RUTA_DATOS):
        return np.array([]), np.array([])

    with open(RUTA_DATOS, newline="", encoding="utf-8") as f:
        for fila in csv.reader(f):
            # Verificamos que la fila tenga el tamaño correcto
            if len(fila) != NUM_CARACTERISTICAS + 1:
                continue
            
            etiqueta = fila[0].upper()
            # FILTRO: Solo aceptamos letras (ignoramos números o basura)
            if etiqueta not in LETRAS_VALIDAS:
                continue

            y.append(etiqueta)
            X.append([float(v) for v in fila[1:]])
            
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    X, y = cargar()

    if len(X) == 0:
        sys.exit("Error: No hay datos válidos en muestras.csv. Ejecuta recolectar.py primero.")

    conteo = Counter(y)
    print("Muestras por letra:", dict(sorted(conteo.items())))

    if len(conteo) < 2:
        sys.exit("Error: Necesitas al menos 2 letras distintas para entrenar el modelo.")

    # Verificamos que ninguna letra tenga muy pocas muestras
    pocas = [k for k, v in conteo.items() if v < MINIMO_POR_SENA]
    if pocas:
        sys.exit(f"Error: Estas letras tienen menos de {MINIMO_POR_SENA} muestras: {pocas}. Graba más.")

    # División 80% entrenamiento, 20% prueba
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nEntrenando el modelo Random Forest (300 árboles)...")
    modelo = RandomForestClassifier(
        n_estimators=300, 
        n_jobs=-1, 
        random_state=42, 
        class_weight="balanced" # Ayuda si hay letras con más muestras que otras
    )
    modelo.fit(X_tr, y_tr)

    # Evaluación del modelo
    pred = modelo.predict(X_te)
    print("\n================ REPORTE DE CLASIFICACIÓN ================")
    print(classification_report(y_te, pred, digits=3))
    print("Letras aprendidas:", list(modelo.classes_))
    
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_te, pred, labels=modelo.classes_))

    # Reentrenamos con TODOS los datos para producción
    print("\nReentrenando con el 100% de los datos...")
    modelo.fit(X, y)
    modelo.n_jobs = 1 # Para que no consuma toda la RAM al usarlo en tiempo real

    # Guardamos el modelo
    joblib.dump({"modelo": modelo, "clases": list(modelo.classes_)}, RUTA_MODELO)
    
    print(f"\n¡Modelo guardado con éxito en {RUTA_MODELO}!")
    print("Ahora puedes ejecutar app.py para probar el reconocimiento en vivo.")


if __name__ == "__main__":
    main()