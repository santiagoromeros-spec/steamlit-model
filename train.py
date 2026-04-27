"""
Entrena los modelos de fatiga y los guarda en modelos.pkl.
Uso: python train.py dataset_ciclismo_fatiga.csv modelos.pkl
"""
import sys
import os
import pandas as pd

# Añadir el directorio padre al path para poder importar modelos
sys.path.insert(0, os.path.dirname(__file__))

from modelos import crear_modelos


def main():
    if len(sys.argv) < 3:
        print("Uso: python train.py <ruta_csv> <ruta_modelo_salida>")
        sys.exit(1)

    csv_path = sys.argv[1]
    modelo_path = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS DE FATIGA EN CICLISMO")
    print("=" * 60)
    print(f"\n📂 Cargando datos desde: {csv_path}")

    modelos = crear_modelos()

    print("🔧 Entrenando modelos...\n")
    resultados, mejor_k = modelos.entrenar(csv_path)

    df_resultados = pd.DataFrame(resultados)

    print("\n" + "=" * 60)
    print("RESULTADOS POR SPLIT")
    print("=" * 60)

    for split_name in ["70/30", "80/20"]:
        print(f"\n🔹 Split {split_name}")
        print("-" * 50)

        # Regresion Lineal
        lr_row = df_resultados[(df_resultados["Modelo"] == "Regresion Lineal") &
                                (df_resultados["Split"] == split_name)].iloc[0]
        print(f"\n  Regresion Lineal:")
        print(f"    MSE: {lr_row['MSE']:.4f}")
        print(f"    R2:  {lr_row['R2']:.4f}")
        print(f"    Intercepto: {lr_row['Intercepto']:.4f}")
        for feat, coef in lr_row["Coeficientes"].items():
            print(f"    {feat}: {coef:.4f}")

        # KNN
        knn_split = df_resultados[(df_resultados["Modelo"] == "KNN") &
                                   (df_resultados["Split"] == split_name)]
        best_knn = knn_split.loc[knn_split["MSE"].idxmin()]

        print(f"\n  KNN (mejor k={int(best_knn['k'])}):")
        print(f"    MSE: {best_knn['MSE']:.4f}")
        print(f"    R2:  {best_knn['R2']:.4f}")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Mejor k para KNN (80/20): {mejor_k}")

    modelos.guardar(modelo_path)
    print(f"\n✅ Modelos guardados en: {modelo_path}")


if __name__ == "__main__":
    main()