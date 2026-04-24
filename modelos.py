"""
Lógica compartida de modelos de regresión para fatiga en ciclismo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

FEATURES = ["frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
            "temperatura", "pendiente", "velocidad"]
TARGET = "fatiga"
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
SPLITS = [(0.3, "70/30"), (0.2, "80/20")]


class ModelosFatiga:
    def __init__(self):
        self.df = None
        self.lr_model = None
        self.knn_model = None
        self.scaler = None
        self.mejor_k = None
        self.resultados = []
        self.entrenado = False

    def cargar_datos(self, path_csv):
        self.df = pd.read_csv(path_csv)
        return self.df.shape

    def entrenar(self, path_csv):
        """Entrena ambos modelos y retorna métricas."""
        self.cargar_datos(path_csv)

        X = self.df[FEATURES]
        y = self.df[TARGET]
        self.resultados = []

        for test_size, split_name in SPLITS:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Regresión Lineal
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)

            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)

            self.resultados.append({
                "Split": split_name,
                "Modelo": "Regresión Lineal",
                "k": "-",
                "MSE": mse_lr,
                "R²": r2_lr,
                "Intercepto": lr.intercept_,
                "Coeficientes": dict(zip(FEATURES, lr.coef_))
            })

            # KNN
            best_k = None
            best_mse = float("inf")
            best_r2 = -float("inf")

            for k in K_VALUES:
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(X_train_scaled, y_train)
                y_pred_knn = knn.predict(X_test_scaled)

                mse_knn = mean_squared_error(y_test, y_pred_knn)
                r2_knn = r2_score(y_test, y_pred_knn)

                self.resultados.append({
                    "Split": split_name,
                    "Modelo": "KNN",
                    "k": k,
                    "MSE": mse_knn,
                    "R²": r2_knn
                })

                if mse_knn < best_mse:
                    best_mse = mse_knn
                    best_r2 = r2_knn
                    best_k = k

            if best_k is not None:
                # Encontrar mejor KNN para este split
                for r in self.resultados:
                    if r["Modelo"] == "KNN" and r["Split"] == split_name and r["k"] == best_k:
                        r["Mejor"] = True
                        break

        # Entrenar modelos finales con 80/20
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_final_scaled = self.scaler.fit_transform(X_train_final)

        # Encontrar mejor k global
        self.mejor_k = None
        mejor_mse_global = float("inf")
        for k in K_VALUES:
            knn_temp = KNeighborsRegressor(n_neighbors=k)
            knn_temp.fit(X_train_final_scaled, y_train_final)
            y_pred_temp = knn_temp.predict(self.scaler.transform(X_test_final))
            mse_temp = mean_squared_error(y_test_final, y_pred_temp)
            if mse_temp < mejor_mse_global:
                mejor_mse_global = mse_temp
                self.mejor_k = k

        # Modelos finales
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_final_scaled, y_train_final)

        self.knn_model = KNeighborsRegressor(n_neighbors=self.mejor_k)
        self.knn_model.fit(X_train_final_scaled, y_train_final)

        self.entrenado = True
        return self.resultados, self.mejor_k

    def predecir(self, datos):
        """Recibe DataFrame con Features, retorna predicciones."""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado. Ejecuta entrenar() primero.")

        datos_scaled = self.scaler.transform(datos[FEATURES])
        pred_lr = self.lr_model.predict(datos_scaled)
        pred_knn = self.knn_model.predict(datos_scaled)

        return pred_lr, pred_knn

    def guardar(self, path):
        """Guarda modelos en pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "lr_model": self.lr_model,
                "knn_model": self.knn_model,
                "scaler": self.scaler,
                "mejor_k": self.mejor_k,
                "resultados": self.resultados,
                "features": FEATURES
            }, f)

    def cargar(self, path):
        """Carga modelos desde pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.lr_model = data["lr_model"]
        self.knn_model = data["knn_model"]
        self.scaler = data["scaler"]
        self.mejor_k = data["mejor_k"]
        self.resultados = data["resultados"]
        self.entrenado = True


def crear_modelos():
    return ModelosFatiga()
