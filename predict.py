"""
Predice fatiga usando modelos ya entrenados.
Uso: python predict.py modelos.pkl
También permite pasar datos como argumentos opcionales.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from modelos import crear_modelos


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict.py <ruta_modelo> [csv_opcional]")
        sys.exit(1)

    modelo_path = sys.argv[1]

    if not os.path.exists(modelo_path):
        print(f"ERROR: No se encontró el archivo: {modelo_path}")
        sys.exit(1)

    print("=" * 60)
    print("PREDICCIÓN DE FATIGA EN CICLISMO")
    print("=" * 60)

    modelos = crear_modelos()
    modelos.cargar(modelo_path)

    print(f"\n✅ Modelos cargados correctamente")
    print(f"   Mejor k para KNN: {modelos.mejor_k}")

    # Datos de ejemplo por defecto
    escenarios = [
        ("Esfuerzo leve (plano)", [120, 200, 80, 30, 20, 2, 30]),
        ("Esfuerzo alto (subida prolongada)", [170, 350, 95, 120, 30, 8, 15]),
        ("Recuperación (bajada)", [90, 150, 70, 10, 15, -3, 40]),
    ]

    print("\n" + "=" * 60)
    print("PREDICCIONES DE FATIGA")
    print("=" * 60)

    for nombre, valores in escenarios:
        datos = pd.DataFrame([valores], columns=[
            "frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
            "temperatura", "pendiente", "velocidad"
        ])
        pred_lr, pred_knn = modelos.predecir(datos)

        print(f"\n🔸 {nombre}:")
        print(f"   Regresión Lineal: {pred_lr[0]:.2f}")
        print(f"   KNN (k={modelos.mejor_k}):    {pred_knn[0]:.2f}")

    # Si se pasa un CSV, se procesa en lote
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\n📂 Procesando CSV: {csv_path}")
            print(f"   Filas: {len(df)}")
            # Asumimos que el CSV tiene las features
            pred_lr, pred_knn = modelos.predecir(df)
            df["pred_lr"] = pred_lr
            df["pred_knn"] = pred_knn
            csv_out = csv_path.replace(".csv", "_predicciones.csv")
            df.to_csv(csv_out, index=False)
            print(f"✅ Guardado en: {csv_out}")
        else:
            print(f"WARNING: CSV no encontrado: {csv_path}")


if __name__ == "__main__":
    main()