"""
Streamlit App — Comparacion de Modelos de Fatiga en Ciclismo.
Estilo: minimalista, retro, blanco y negro, sin emojis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

from modelos import crear_modelos

# ─── Configuracion de pagina ───
st.set_page_config(
    page_title="FATIGA // Modelo de Ciclismo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Global — Monospace Retro ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    :root {
        --black: #000000;
        --white: #ffffff;
        --gray-dark: #222222;
        --gray-mid: #555555;
        --gray-light: #aaaaaa;
        --gray-border: #cccccc;
    }

    * {
        font-family: 'Space Mono', 'Courier New', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 2px solid #ffffff;
        width: 260px !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }

    /* Main background */
    .stApp {
        background-color: #ffffff;
    }

    /* Headers */
    h1, h2, h3, h4 {
        font-family: 'Space Mono', monospace !important;
        color: #000000 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        color: #555555;
        border-bottom: 2px solid #000000;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 3px solid #000000;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Space Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem;
        border: 2px solid #000000;
        background-color: #000000;
        color: #ffffff;
        border-radius: 0;
        width: 100%;
        transition: all 0.1s;
    }

    .stButton > button:hover {
        background-color: #ffffff;
        color: #000000;
    }

    /* Secondary button outline */
    .stButton > button[kind="secondary"] {
        background-color: #ffffff;
        color: #000000;
    }

    /* Metric containers */
    [data-testid="stMetric"] {
        border: 1px solid #000000;
        padding: 16px;
        background: #ffffff;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        color: #555555;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #000000;
    }

    /* Tables */
    .stDataFrame [data-testid="stTable"] {
        border: 1px solid #000000;
        font-family: 'Space Mono', monospace;
    }

    table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'Space Mono', monospace;
    }

    th {
        background-color: #000000;
        color: #ffffff;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        padding: 8px;
        border: 1px solid #000000;
    }

    td {
        border: 1px solid #cccccc;
        padding: 8px;
        font-size: 0.8rem;
        color: #000000;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #000000;
        margin: 2rem 0;
    }

    /* Success/Warning/Error boxes */
    .stAlert {
        border-radius: 0;
        border-left: 4px solid #000000;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        border: 1px solid #000000;
        background: #ffffff;
    }

    .streamlit-expanderContent {
        border: 1px solid #cccccc;
        border-top: none;
        background: #ffffff;
    }

    /* Selectbox / Number inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        font-family: 'Space Mono', monospace;
        border: 1px solid #000000;
        border-radius: 0;
    }

    /* Download button */
    .stDownloadButton > button {
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        border: 2px solid #000000;
        background: #ffffff;
        color: #000000;
        border-radius: 0;
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #000000;
        border-radius: 0;
        background: #ffffff;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 4rem;
        font-family: 'Space Mono', monospace;
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #000000;
        border-bottom: 2px solid #000000;
        padding-bottom: 4px;
        margin-bottom: 1.5rem;
        margin-top: 2.5rem;
    }

    /* Prediction result box */
    .pred-box {
        border: 2px solid #000000;
        padding: 16px 24px;
        text-align: center;
        margin-top: 1rem;
    }

    .pred-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #555555;
    }

    .pred-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        line-height: 1;
    }

    /* Fatiga level */
    .fatiga-level {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 700;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ─── Archivos ───
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "dataset_ciclismo_fatiga.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelos.pkl")

# ─── Estado de sesion ───
if "modelos" not in st.session_state:
    st.session_state.modelos = None
if "entrenado" not in st.session_state:
    st.session_state.entrenado = False


def cargar_o_entrenar(csv_path, model_path):
    modelos = crear_modelos()
    if os.path.exists(model_path):
        modelos.cargar(model_path)
        return modelos, False
    else:
        resultados, mejor_k = modelos.entrenar(csv_path)
        modelos.guardar(model_path)
        return modelos, True


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
        <div style="color:#ffffff; padding-top:1rem;">
            <div style="font-size:1.8rem; font-weight:700; letter-spacing:0.3em; line-height:1;">
                FATIGA
            </div>
            <div style="font-size:0.6rem; letter-spacing:0.2em; color:#888888; margin-top:4px;">
                MODELO PREDICTIVO // CICLISMO
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-top:1px solid #333; margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Botones sidebar
    btn_entrenar = st.button("ENTRENAR / REENTRENAR")
    btn_ver_datos = st.button("VER DATOS")

    # Estado del modelo
    st.markdown("<hr style='border-top:1px solid #333; margin:1.5rem 0;'>", unsafe_allow_html=True)

    if st.session_state.entrenado:
        estado_modelo = "LISTO"
        estado_color = "#27ae60"
    else:
        estado_modelo = "SIN ENTRENAR"
        estado_color = "#888888"

    st.markdown(f"""
        <div style="color:#aaaaaa; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em;">
            ESTADO DEL MODELO
        </div>
        <div style="color:{estado_color}; font-size:0.9rem; font-weight:700; letter-spacing:0.2em; margin-top:4px;">
            {estado_modelo}
        </div>
    """, unsafe_allow_html=True)

    # Acciones al entrenar
    if btn_entrenar:
        with st.spinner("Entrenando..."):
            modelos, fue_entrenado = cargar_o_entrenar(CSV_PATH, MODEL_PATH)
            st.session_state.modelos = modelos
            st.session_state.entrenado = True
        if fue_entrenado:
            st.success("Modelo entrenado y guardado.")
        else:
            st.info("Modelo cargado desde archivo.")

    if btn_ver_datos:
        if os.path.exists(CSV_PATH):
            df_temp = pd.read_csv(CSV_PATH)
            st.dataframe(df_temp.head(20))
            st.caption(f"Registros: {len(df_temp)}")
        else:
            st.error("CSV no encontrado.")

    # Auto-cargar
    if not st.session_state.entrenado and os.path.exists(MODEL_PATH):
        try:
            modelos = crear_modelos()
            modelos.cargar(MODEL_PATH)
            st.session_state.modelos = modelos
            st.session_state.entrenado = True
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">// METRICAS DE EVALUACION</p>', unsafe_allow_html=True)

if st.session_state.entrenado:
    modelos = st.session_state.modelos
    df_res = pd.DataFrame(modelos.resultados)
    df_res["MSE"] = pd.to_numeric(df_res["MSE"], errors="coerce")
    df_res["R2"] = pd.to_numeric(df_res["R2"], errors="coerce")

    tab1, tab2 = st.tabs(["SPLIT 70/30", "SPLIT 80/20"])

    for tab, split_name in zip([tab1, tab2], ["70/30", "80/20"]):
        with tab:
            df_split = df_res[df_res["Split"] == split_name].copy()

            # Regresion Lineal
            lr_row = df_split[df_split["Modelo"] == "Regresion Lineal"].iloc[0]

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("REGRESION LINEAL", f"MSE {lr_row['MSE']:.4f}", f"R2 {lr_row['R2']:.4f}")

            knn_split = df_split[df_split["Modelo"] == "KNN"]
            best_knn_idx = knn_split["MSE"].idxmin()
            best_knn = knn_split.loc[best_knn_idx]

            with col_m2:
                st.metric(f"KNN (K={int(best_knn['k'])})", f"MSE {best_knn['MSE']:.4f}", f"R2 {best_knn['R2']:.4f}")

            # Tabla de KNN por k
            st.markdown("---")
            st.markdown("**KNN — VALORES DE K**")

            knn_rows = knn_split.copy()
            knn_rows["k"] = knn_rows["k"].astype(int)
            knn_rows = knn_rows.sort_values("k")
            knn_rows["MSE"] = knn_rows["MSE"].apply(lambda x: f"{x:.4f}")
            knn_rows["R2"] = knn_rows["R2"].apply(lambda x: f"{x:.4f}")

            # Marcamos el mejor
            knn_display = knn_rows[["k", "MSE", "R2"]].rename(
                columns={"k": "K", "MSE": "MSE", "R2": "R2"}
            )
            st.dataframe(knn_display, hide_index=True, use_container_width=True)

            # Coeficientes LR (solo en 80/20)
            if split_name == "80/20":
                coefs = lr_row.get("Coeficientes", {})
                if coefs and isinstance(coefs, dict):
                    with st.expander("VER COEFICIENTES — REGRESION LINEAL"):
                        coef_df = pd.DataFrame([
                            {"Feature": k, "Coeficiente": f"{v:.4f}"}
                            for k, v in coefs.items()
                        ])
                        st.dataframe(coef_df, hide_index=True, use_container_width=True)

    # Comparativa
    st.markdown("---")
    st.markdown('<p class="section-title">// COMPARATIVA 80/20</p>', unsafe_allow_html=True)

    lr_best = df_res[(df_res["Modelo"] == "Regresion Lineal") & (df_res["Split"] == "80/20")].iloc[0]
    knn_best = df_res[(df_res["Modelo"] == "KNN") & (df_res["Split"] == "80/20")].loc[
        df_res[(df_res["Modelo"] == "KNN") & (df_res["Split"] == "80/20")]["MSE"].idxmin()
    ]

    cc1, cc2, cc3 = st.columns([1, 1, 1])

    with cc1:
        st.metric("REGRESION LINEAL", f"MSE {lr_best['MSE']:.4f}", f"R2 {lr_best['R2']:.4f}")
    with cc2:
        st.metric(f"KNN K={int(knn_best['k'])}", f"MSE {knn_best['MSE']:.4f}", f"R2 {knn_best['R2']:.4f}")

    if lr_best["MSE"] < knn_best["MSE"]:
        mejor_nombre = "REGRESION LINEAL"
        mejor_mse = lr_best["MSE"]
    else:
        mejor_nombre = f"KNN K={int(knn_best['k'])}"
        mejor_mse = knn_best["MSE"]

    with cc3:
        st.markdown(f"""
            <div style="border:2px solid #000; padding:16px; text-align:center;">
                <div style="font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em; color:#555;">
                    MEJOR MODELO
                </div>
                <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:8px;">
                    {mejor_nombre}
                </div>
                <div style="font-size:0.7rem; color:#555; margin-top:4px;">
                    MSE {mejor_mse:.4f}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # PREDICCION
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">// PREDICCION</p>', unsafe_allow_html=True)

    escenarios = {
        "ESFUERZO LEVE (PLANO)":       [120, 200, 80, 30,  20,  2,  30],
        "ESFUERZO MODERADO (SUBIDA SUAVE)": [150, 280, 88, 60, 25, 5, 22],
        "ESFUERZO ALTO (SUBIDA PROLONGADA)": [170, 350, 95, 120, 30, 8, 15],
        "RECUPERACION (BAJADA)":       [90,  150, 70, 10,  15, -3, 40],
        "PERSONALIZADO": None,
    }

    escenario = st.selectbox("ESCENARIO", list(escenarios.keys()))

    if escenario == "PERSONALIZADO":
        col1, col2, col3 = st.columns(3)
        with col1:
            fc  = st.number_input("FREC. CARDIACA (lpm)",  min_value=50,  max_value=200, value=120, step=1)
            pot = st.number_input("POTENCIA (W)",          min_value=50,  max_value=500, value=200, step=5)
            cad = st.number_input("CADENCIA (rpm)",          min_value=30,  max_value=130, value=80,  step=1)
        with col2:
            t   = st.number_input("TIEMPO (min)",           min_value=1,   max_value=300, value=30,  step=1)
            tmp = st.number_input("TEMPERATURA (C)",        min_value=-10, max_value=50,  value=20,  step=1)
            pen = st.number_input("PENDIENTE (°)",          min_value=-15, max_value=20,  value=2,   step=1)
        with col3:
            vel = st.number_input("VELOCIDAD (km/h)",        min_value=5,   max_value=80,  value=30,  step=1)
        valores = [fc, pot, cad, t, tmp, pen, vel]
    else:
        valores = escenarios[escenario]

    if st.button("PREDECIR FATIGA"):
        datos = pd.DataFrame([valores], columns=[
            "frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
            "temperatura", "pendiente", "velocidad"
        ])
        pred_lr, pred_knn = modelos.predecir(datos)

        fatiga_prom = (pred_lr[0] + pred_knn[0]) / 2
        if fatiga_prom < 40:
            nivel = "BAJO"
        elif fatiga_prom < 60:
            nivel = "MODERADO"
        else:
            nivel = "ALTO"

        col_r, col_k, col_f = st.columns(3)

        with col_r:
            st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">REGRESION LINEAL</div>
                    <div class="pred-value">{pred_lr[0]:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_k:
            st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">KNN K={modelos.mejor_k}</div>
                    <div class="pred-value">{pred_knn[0]:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_f:
            st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">NIVEL DE FATIGA</div>
                    <div class="pred-value fatiga-level">{nivel}</div>
                    <div class="pred-label" style="margin-top:8px;">PROMEDIO {fatiga_prom:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        # CSV batch
        st.markdown("---")
        with st.expander("PREDICCION DESDE ARCHIVO CSV"):
            uploaded = st.file_uploader(
                "SUBIR CSV",
                type=["csv"],
                label_visibility="collapsed"
            )
            if uploaded:
                df_up = pd.read_csv(uploaded)
                FEATURES = ["frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
                            "temperatura", "pendiente", "velocidad"]
                faltantes = [c for c in FEATURES if c not in df_up.columns]
                if faltantes:
                    st.error(f"Columnas faltantes: {faltantes}")
                else:
                    pred_lr, pred_knn = modelos.predecir(df_up)
                    df_up["pred_regresion_lineal"] = pred_lr
                    df_up["pred_knn"] = pred_knn
                    st.dataframe(df_up, hide_index=True, use_container_width=True)
                    csv_descarga = df_up.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "DESCARGAR PREDICCIONES",
                        csv_descarga,
                        "predicciones_fatiga.csv",
                        "text/csv"
                    )

else:
    st.info("Usa el boton ENTRAR del panel izquierdo para comenzar.")

# ─── FOOTER ───
st.markdown('<p class="footer">// FATIGA — MODELO DE PREDICCION EN CICLISMO //</p>', unsafe_allow_html=True)
