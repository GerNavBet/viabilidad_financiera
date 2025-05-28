import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Título ---
st.title("Asistente de Viabilidad Económico-Financiera")
st.write("Introduce los datos básicos de tu plan de negocio y obtén una estimación de su viabilidad.")

# --- Formulario de entrada ---
with st.form("viabilidad_form"):
    inversion_inicial = st.number_input("Inversión inicial prevista (€)", min_value=0, value=25000)
    ventas_mensuales = st.number_input("Ventas mensuales estimadas (€)", min_value=0, value=7000)
    costes_fijos = st.number_input("Costes fijos mensuales (€)", min_value=0, value=2000)
    margen_bruto = st.slider("Margen bruto (%)", 0, 100, 60)
    meses_recuperacion = st.number_input("Plazo estimado para recuperar la inversión (meses)", min_value=1, max_value=60, value=24)
    experiencia_sector = st.selectbox("Experiencia previa en el sector", ["Alta", "Media", "Baja"])
    estructura_coste = st.selectbox("Estructura de costes definida", ["Sí", "No"])

    submitted = st.form_submit_button("Calcular Viabilidad")

# --- Preparar datos sintéticos y modelo de regresión ---
def entrenar_modelo_dummy():
    np.random.seed(0)
    X = np.random.rand(100, 6)
    y = (X[:, 1] * 10000 - X[:, 2] * 3000 + X[:, 3] * 100 > 5000).astype(int)  # Generar target viable/no viable
    model = LogisticRegression()
    model.fit(X, y)
    scaler = StandardScaler()
    scaler.fit(X)
    return model, scaler

modelo, scaler = entrenar_modelo_dummy()

# --- Codificación de variables categóricas ---
def codificar_inputs(inversion, ventas, fijos, margen, recuperacion, experiencia, estructura):
    exp_map = {"Baja": 0, "Media": 0.5, "Alta": 1}
    estructura_map = {"No": 0, "Sí": 1}
    return [inversion / 100000, ventas / 10000, fijos / 10000, margen / 100, recuperacion / 60, exp_map[experiencia], estructura_map[estructura]]

# --- Predicción de viabilidad ---
if submitted:
    entrada = np.array([codificar_inputs(
        inversion_inicial, ventas_mensuales, costes_fijos, margen_bruto,
        meses_recuperacion, experiencia_sector, estructura_coste
    )])

    entrada_escalada = scaler.transform(entrada[:, :6])
    prediccion = modelo.predict(entrada_escalada)[0]
    probabilidad = modelo.predict_proba(entrada_escalada)[0][1]

    st.subheader("Resultado de Viabilidad")
    if prediccion:
        st.success(f"Plan de negocio potencialmente viable (probabilidad: {probabilidad:.2%})")
    else:
        st.error(f"Plan con baja probabilidad de viabilidad (probabilidad: {probabilidad:.2%})")
    st.write("Esta estimación se basa en un modelo de referencia. Para una evaluación personalizada, recomendamos una sesión de análisis financiero.")
