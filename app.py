"""
Streamlit App — Comparación de Modelos de Fatiga en Ciclismo.
Se entrenan bajo demanda y se muestran métricas y predicciones.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

from modelos import crear_modelos

# ─── Configuración de página ───
st.set_page_config(
    page_title="Fatiga Ciclista — Comparación de Modelos",
    page_icon="🚴",
    layout="centered",
)

# ─── Archivos ───
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "dataset_ciclismo_fatiga.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelos.pkl")

# ─── Inicializar estado ───
if "modelos" not in st.session_state:
    st.session_state.modelos = None
if "entrenado" not in st.session_state:
    st.session_state.entrenado = False

# ─── Funciones auxiliares ───
def cargar_o_entrenar(csv_path, model_path):
    """Carga modelos desde pickle o entrena si no existen."""
    modelos = crear_modelos()

    if os.path.exists(model_path):
        modelos.cargar(model_path)
        return modelos, False  # cargado, no entrenado
    else:
        resultados, mejor_k = modelos.entrenar(csv_path)
        modelos.guardar(model_path)
        return modelos, True  # entrenado


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f77b4; text-align: center; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-top: 1.5rem; }
    .metric-card { background: #f0f8ff; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
    .best-badge { background: #27ae60; color: white; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; margin-left: 8px; }
    .footer { text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚴 Modelo de Fatiga en Ciclismo</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#666;'>Comparación entre <strong>Regresión Lineal</strong> "
    "y <strong>KNN (K-Nearest Neighbors)</strong></p>",
    unsafe_allow_html=True
)
st.divider()

# ════════════════════════════════════════════════════════════
# 1 — ENTRENAMIENTO BAJO DEMANDA
# ════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">📊 Entrenamiento de Modelos</p>', unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    btn_entrenar = st.button("🔧 Entrenar / Reentrenar modelos", type="primary", use_container_width=True)

with col_btn2:
    ver_datos = st.button("📂 Ver datos cargados", use_container_width=True)

# Entrenar o recargar
if btn_entrenar:
    with st.spinner("Entrenando modelos..."):
        modelos, fue_entrenado = cargar_o_entrenar(CSV_PATH, MODEL_PATH)
        st.session_state.modelos = modelos
        st.session_state.entrenado = True
        if fue_entrenado:
            st.success("✅ Modelos entrenados y guardados en `modelos.pkl`")
        else:
            st.info("ℹ️ Modelos cargados desde archivo existente")

if ver_datos:
    if os.path.exists(CSV_PATH):
        df_temp = pd.read_csv(CSV_PATH)
        st.dataframe(df_temp.head(20), use_container_width=True)
        st.caption(f"Total de registros: {len(df_temp)}")
    else:
        st.error("❌ No se encontró el archivo CSV.")

# Auto-cargar si ya está entrenado
if not st.session_state.entrenado and os.path.exists(MODEL_PATH):
    try:
        modelos = crear_modelos()
        modelos.cargar(MODEL_PATH)
        st.session_state.modelos = modelos
        st.session_state.entrenado = True
        st.info("📦 Modelos cargados automáticamente (archivo existente).")
    except Exception:
        pass

# ════════════════════════════════════════════════════════════
# 2 — MÉTRICAS (solo si está entrenado)
# ════════════════════════════════════════════════════════════
if st.session_state.entrenado:
    modelos = st.session_state.modelos
    df_res = pd.DataFrame(modelos.resultados)

    # Limpiar filas sin MSE válido
    df_res["MSE"] = pd.to_numeric(df_res["MSE"], errors="coerce")
    df_res["R²"] = pd.to_numeric(df_res["R²"], errors="coerce")

    st.markdown("---")
    st.markdown('<p class="section-title">📈 Métricas de Evaluación</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 70 / 30", "📊 80 / 20"])

    for tab, split_name in zip(tabs, ["70/30", "80/20"]):
        with tab:
            df_split = df_res[df_res["Split"] == split_name].copy()

            # — Regresión Lineal —
            lr_row = df_split[df_split["Modelo"] == "Regresión Lineal"].iloc[0]

            st.markdown(f"""
            <div class="metric-card">
                <strong>Regresión Lineal</strong><br>
                <span>MSE: <strong>{lr_row["MSE"]:.4f}</strong></span> &nbsp;|&nbsp;
                <span>R²: <strong>{lr_row["R²"]:.4f}</strong></span>
            </div>
            """, unsafe_allow_html=True)

            # Coeficientes
            coefs = lr_row.get("Coeficientes", {})
            if coefs and isinstance(coefs, dict) and split_name == "80/20":
                with st.expander("🔢 Ver coeficientes del modelo"):
                    coef_df = pd.DataFrame([
                        {"Feature": k, "Coeficiente": f"{v:.4f}"}
                        for k, v in coefs.items()
                    ])
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # — Tabla de KNN —
            st.markdown("**KNN — Valores de k:**")
            knn_rows = df_split[df_split["Modelo"] == "KNN"].copy()
            knn_rows["k"] = knn_rows["k"].astype(int)

            # Resaltar mejor k
            best_idx = knn_rows["MSE"].idxmin()
            knn_rows["Mejor"] = knn_rows.index.map(
                lambda i: "⭐ Mejor" if i == best_idx else ""
            )

            def highlight_best(s):
                return ["background-color: #d4edda; font-weight: 600" if v == "⭐ Mejor" else "" for v in s]

            st.dataframe(
                knn_rows[["k", "MSE", "R²", "Mejor"]].rename(
                    columns={"k": "k", "MSE": "MSE", "R²": "R²", "Mejor": ""}
                ),
                use_container_width=True,
                hide_index=True
            )

            # Mejor KNN resumido
            best_row = knn_rows.loc[best_idx]
            st.success(f"⭐ Mejor KNN: k = {int(best_row['k'])} → "
                       f"MSE = {best_row['MSE']:.4f}  |  R² = {best_row['R²']:.4f}")

    # — Comparativa lado a lado —
    st.markdown("---")
    st.markdown("### 📊 Comparación 80/20 — Mejor de cada modelo")

    lr_best = df_res[(df_res["Modelo"] == "Regresión Lineal") & (df_res["Split"] == "80/20")].iloc[0]
    knn_best = df_res[(df_res["Modelo"] == "KNN") & (df_res["Split"] == "80/20")].loc[
        df_res[(df_res["Modelo"] == "KNN") & (df_res["Split"] == "80/20")]["MSE"].idxmin()
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Regresión Lineal (80/20)", f"MSE: {lr_best['MSE']:.4f}", f"R²: {lr_best['R²']:.4f}")
    with c2:
        st.metric(f"KNN k={int(knn_best['k'])} (80/20)", f"MSE: {knn_best['MSE']:.4f}", f"R²: {knn_best['R²']:.4f}")

    # Determinar el mejor modelo
    if lr_best["MSE"] < knn_best["MSE"]:
        mejor_nombre = "Regresión Lineal"
        mejor_mse = lr_best["MSE"]
    else:
        mejor_nombre = f"KNN (k={int(knn_best['k'])})"
        mejor_mse = knn_best["MSE"]

    st.markdown(f"""
    <div style="background:#d4edda; border-radius:8px; padding:12px 16px; text-align:center; font-size:1.1rem;">
        🏆 <strong>Mejor modelo general:</strong> {mejor_nombre} (MSE: {mejor_mse:.4f})
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # 3 — PREDICCIONES
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="section-title">🔮 Predicción con Nuevos Datos</p>', unsafe_allow_html=True)

    st.markdown("##### Escoge un escenario predefinido o ingresa tus propios datos:")

    escenario_opciones = {
        "Esfuerzo leve (plano)":        [120, 200, 80, 30,  20,  2,  30],
        "Esfuerzo moderado (subida suave)": [150, 280, 88, 60, 25, 5, 22],
        "Esfuerzo alto (subida prolongada)": [170, 350, 95, 120, 30, 8, 15],
        "Recuperación (bajada)":        [90,  150, 70, 10,  15, -3, 40],
        "Personalizado (escribe abajo)": None,
    }

    escenario = st.selectbox("Escenario:", list(escenario_opciones.keys()))

    if escenario == "Personalizado (escribe abajo)":
        st.markdown("##### Ingresa los valores:")
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                fc  = st.number_input("Frecuencia cardíaca (lpm)",  min_value=50,  max_value=200, value=120, step=1)
                pot = st.number_input("Potencia (W)",               min_value=50,  max_value=500, value=200, step=5)
                cad = st.number_input("Cadencia (rpm)",             min_value=30,  max_value=130, value=80,  step=1)
            with c2:
                t   = st.number_input("Tiempo (min)",              min_value=1,   max_value=300, value=30,  step=1)
                tmp = st.number_input("Temperatura (°C)",           min_value=-10, max_value=50,  value=20,  step=1)
                pen = st.number_input("Pendiente (°)",             min_value=-15, max_value=20,  value=2,   step=1)
            with c3:
                vel = st.number_input("Velocidad (km/h)",           min_value=5,   max_value=80,  value=30,  step=1)

        valores = [fc, pot, cad, t, tmp, pen, vel]

    else:
        valores = escenario_opciones[escenario]

    # Botón predecir
    if st.button("🔮 Predecir Fatiga", type="primary"):
        datos = pd.DataFrame([valores], columns=[
            "frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
            "temperatura", "pendiente", "velocidad"
        ])

        with st.spinner("Prediciendo..."):
            pred_lr, pred_knn = modelos.predecir(datos)

        st.markdown("---")
        st.markdown("### 📋 Resultado de Predicción")
        st.dataframe(datos.T.rename(columns={0: "Valor"}), use_container_width=True)

        col_r, col_k = st.columns(2)
        with col_r:
            st.metric("🔵 Regresión Lineal", f"{pred_lr[0]:.2f}", help="Predicción con modelo lineal")
        with col_k:
            st.metric("🟠 KNN (entrenado con mejores datos)", f"{pred_knn[0]:.2f}", help=f"Modelo KNN con k={modelos.mejor_k}")

        # Indicador general
        fatiga_prom = (pred_lr[0] + pred_knn[0]) / 2
        if fatiga_prom < 40:
            nivel = "🟢 Fatiga baja"
            color = "#27ae60"
        elif fatiga_prom < 60:
            nivel = "🟡 Fatiga moderada"
            color = "#f39c12"
        else:
            nivel = "🔴 Fatiga alta"
            color = "#e74c3c"

        st.markdown(f"""
        <div style="background:{color}; color:white; border-radius:8px; padding:12px 16px; text-align:center; font-size:1rem;">
            <strong>{nivel}</strong> — Promedio: {fatiga_prom:.2f}
        </div>
        """, unsafe_allow_html=True)

    # ─── Predicción por CSV ───
    st.markdown("---")
    with st.expander("📂 Predecir desde archivo CSV"):
        uploaded = st.file_uploader("Sube un CSV con las columnas: frecuencia_cardiaca, potencia, cadencia, tiempo, temperatura, pendiente, velocidad", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            FEATURES = ["frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
                        "temperatura", "pendiente", "velocidad"]
            # Validar columnas
            faltantes = [c for c in FEATURES if c not in df_up.columns]
            if faltantes:
                st.error(f"Columnas faltantes: {faltantes}")
            else:
                pred_lr, pred_knn = modelos.predecir(df_up)
                df_up["pred_regresion_lineal"] = pred_lr
                df_up["pred_knn"] = pred_knn
                st.dataframe(df_up, use_container_width=True)
                csv_descarga = df_up.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar predicciones CSV", csv_descarga,
                                  "predicciones_fatiga.csv", "text/csv")

else:
    st.info("👈 Usa el botón **'Entrenar / Reentrenar modelos'** para comenzar.")

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown('<p class="footer">Modelo de Fatiga en Ciclismo — Herramienta de comparación de modelos</p>', unsafe_allow_html=True)