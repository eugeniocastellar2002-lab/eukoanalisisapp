import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Tipster Pro Analytics Terminal",
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ Terminal Avanzada de Análisis Estadístico para Apuestas Deportivas")
st.caption("Herramienta Profesional de Modelación Cuantitativa | Tipster Pro Edition")

# --- DATASETS POR DEFECTO (Millonarios vs Deportivo Cali) ---
default_a = {
    'Fecha': ['06/09/2026', '02/09/2026', '30/08/2026', '27/08/2026', '22/08/2026'],
    'GF': [1, 2, 0, 2, 1],
    'GC': [1, 3, 0, 0, 1],
    'Tiros': [12, 14, 16, 11, 10],
    'Tiros_Rival': [10, 11, 6, 8, 9],
    'TA': [4, 5, 6, 4, 3],
    'Corneres': [5, 6, 8, 4, 5],
    'Faltas': [14, 16, 11, 13, 15],
    'Tarjetas': [1, 5, 2, 3, 4]
}

default_b = {
    'Fecha': ['07/09/2026', '30/08/2026', '26/08/2026', '23/08/2026', '08/08/2026'],
    'GF': [0, 1, 1, 2, 2],
    'GC': [3, 1, 2, 2, 0],
    'Tiros': [7, 13, 9, 15, 11],
    'Tiros_Rival': [15, 9, 14, 10, 7],
    'TA': [2, 4, 3, 5, 4],
    'Corneres': [3, 6, 4, 7, 5],
    'Faltas': [17, 14, 18, 12, 15],
    'Tarjetas': [3, 6, 3, 2, 4]
}

# --- SIDEBAR: GESTIÓN Y CÓDIGO DE DATOS ---
st.sidebar.header("⚙️ Configuración y Datos")
equipo_a_name = st.sidebar.text_input("Nombre Equipo A", "Millonarios FC")
equipo_b_name = st.sidebar.text_input("Nombre Equipo B", "Deportivo Cali")

modo_datos = st.sidebar.radio("Fuente de Datos", ["Datos Predeterminados", "Cargar Archivo (CSV/Excel)"])

df_a = pd.DataFrame(default_a)
df_b = pd.DataFrame(default_b)

if modo_datos == "Cargar Archivo (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo con las pestañas o columnas de cada equipo", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                excel_sheets = pd.ExcelFile(uploaded_file)
                sheet_names = excel_sheets.sheet_names
                sheet_a = st.sidebar.selectbox("Hoja Equipo A", sheet_names, index=0)
                sheet_b = st.sidebar.selectbox("Hoja Equipo B", sheet_names, index=min(1, len(sheet_names)-1))
                df_a = pd.read_excel(uploaded_file, sheet_name=sheet_a)
                df_b = pd.read_excel(uploaded_file, sheet_name=sheet_b)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
                st.sidebar.info("CSV cargado para ambos equipos.")
                df_a = df_uploaded
                df_b = df_uploaded
        except Exception as e:
            st.sidebar.error(f"Error al leer el archivo: {e}")

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptiva Avanzada", 
    "🎲 Calculadora Poisson", 
    "📈 Regresión Lineal & ANOVA", 
    "🔔 Probabilidad Normal", 
    "💰 Calculadora EV & Stake"
])

# --- TAB 1: ESTADÍSTICA DESCRIPTIVA ---
with tab1:
    st.subheader("Tabla de Estadísticos Descriptivos Completos")
    col1, col2 = st.columns(2)
    
    def get_stats_table(df):
        numeric_df = df.select_dtypes(include=[np.number])
        stats_dict = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) == 0:
                continue
            mean = series.mean()
            std = series.std() if len(series) > 1 else 0
            cv = (std / mean * 100) if mean != 0 else 0
            skew_val = stats.skew(series) if len(series) > 2 else 0
            skew_std = skew_val / np.sqrt(6 / len(series)) if len(series) > 0 else 0
            
            stats_dict[col] = {
                'Recuento': int(series.count()),
                'Promedio': round(mean, 2),
                'Desv. Estándar': round(std, 2),
                'Coef. Variación (%)': round(cv, 2),
                'Mínimo': series.min(),
                'Máximo': series.max(),
                'Sesgo Estandarizado': round(skew_std, 2)
            }
        return pd.DataFrame(stats_dict).T

    with col1:
        st.write(f"### {equipo_a_name}")
        st.dataframe(get_stats_table(df_a), use_container_width=True)
        
    with col2:
        st.write(f"### {equipo_b_name}")
        st.dataframe(get_stats_table(df_b), use_container_width=True)

# --- TAB 2: POISSON ---
with tab2:
    st.subheader("Modelo de Distribución de Poisson para Marcador Correcto")
    col_p1, col_p2 = st.columns(2)
    
    default_lambda_a = float(df_a['GF'].mean()) if 'GF' in df_a.columns else 1.2
    default_lambda_b = float(df_b['GF'].mean()) if 'GF' in df_b.columns else 1.0
    
    with col_p1:
        lambda_a = st.number_input(f"Expected Goals (xG) {equipo_a_name}", value=default_lambda_a, step=0.1)
    with col_p2:
        lambda_b = st.number_input(f"Expected Goals (xG) {equipo_b_name}", value=default_lambda_b, step=0.1)
        
    max_goals = 5
    poisson_a = [stats.poisson.pmf(i, lambda_a) for i in range(max_goals + 1)]
    poisson_b = [stats.poisson.pmf(i, lambda_b) for i in range(max_goals + 1)]
    
    matrix = np.outer(poisson_a, poisson_b)
    
    fig_poisson = px.imshow(
        matrix, 
        labels=dict(x=f"Goles {equipo_b_name}", y=f"Goles {equipo_a_name}", color="Probabilidad"),
        x=[str(i) for i in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        text_auto=".2%",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_poisson, use_container_width=True)
    
    p_win_a = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_win_b = np.triu(matrix, 1).sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Victoria {equipo_a_name}", f"{p_win_a:.1%}")
    c2.metric("Empate", f"{p_draw:.1%}")
    c3.metric(f"Victoria {equipo_b_name}", f"{p_win_b:.1%}")

# --- TAB 3: REGRESIÓN LINEAL & ANOVA ---
with tab3:
    st.subheader("Modelo de Regresión Lineal & Análisis ANOVA")
    
    combined_df = pd.concat([df_a, df_b], ignore_index=True)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col_r1, col_r2 = st.columns(2)
        x_var = col_r1.selectbox("Variable Independiente (X)", numeric_cols, index=min(2, len(numeric_cols)-1))
        y_var = col_r2.selectbox("Variable Dependiente (Y)", numeric_cols, index=0)
        
        clean_df = combined_df[[x_var, y_var]].dropna()
        X = clean_df[x_var]
        Y = clean_df[y_var]
        
        if len(X) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
            
            fig_reg = px.scatter(clean_df, x=x_var, y=y_var, trendline="ols", title=f"Regresión: {y_var} vs {x_var} (R² = {r_value**2:.3f})")
            st.plotly_chart(fig_reg, use_container_width=True)
            
            st.write("### Tabla de Análisis de Varianza (ANOVA)")
            y_hat = intercept + slope * X
            ss_tot = np.sum((Y - np.mean(Y))**2)
            ss_reg = np.sum((y_hat - np.mean(Y))**2)
            ss_res = np.sum((Y - y_hat)**2)
            
            df_reg = 1
            df_res = len(Y) - 2
            ms_reg = ss_reg / df_reg
            ms_res = ss_res / df_res if df_res > 0 else 0000.1
            f_stat = ms_reg / ms_res if ms_res > 0 else 0
            p_val_f = stats.f.sf(f_stat, df_reg, df_res) if df_res > 0 else 1
            
            anova_df = pd.DataFrame({
                'Fuente de Variación': ['Regresión', 'Residuos', 'Total'],
                'Suma de Cuadrados (SS)': [round(ss_reg, 3), round(ss_res, 3), round(ss_tot, 3)],
                'Grados de Libertad (DF)': [df_reg, df_res, len(Y) - 1],
                'Media Cuadrática (MS)': [round(ms_reg, 3), round(ms_res, 3), '-'],
                'F-Stat': [round(f_stat, 3), '-', '-'],
                'p-valor': [round(p_val_f, 4), '-', '-']
            })
            st.dataframe(anova_df, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 3 registros con datos numéricos para calcular la regresión.")

# --- TAB 4: PROBABILIDAD NORMAL ---
with tab4:
    st.subheader("Distribución Normal para Mercados de Líneas / Overs / Unders")
    
    if len(numeric_cols) > 0:
        metric_selected = st.selectbox("Métrica a Evaluar", numeric_cols, index=0)
        linea = st.number_input("Línea del Mercado (Ejemplo: 9.5 Córneres / 2.5 Goles)", value=9.5, step=0.5)
        
        series_data = combined_df[metric_selected].dropna()
        mu = series_data.mean()
        sigma = series_data.std()
        
        if sigma > 0:
            prob_under = stats.norm.cdf(linea, mu, sigma)
            prob_over = 1 - prob_under
            
            st.write(f"**Media ($\mu$):** {mu:.2f} | **Desviación Estándar ($\sigma$):** {sigma:.2f}")
            
            col_n1, col_n2 = st.columns(2)
            col_n1.metric(f"Probabilidad UNDER {linea}", f"{prob_under:.2%}")
            col_n2.metric(f"Probabilidad OVER {linea}", f"{prob_over:.2%}")
            
            x_axis = np.linspace(max(0, mu - 4*sigma), mu + 4*sigma, 100)
            y_axis = stats.norm.pdf(x_axis, mu, sigma)
            
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='Curva Normal'))
            fig_norm.add_vline(x=linea, line_dash="dash", line_color="red", annotation_text=f"Línea {linea}")
            st.plotly_chart(fig_norm, use_container_width=True)
        else:
            st.warning("La desviación estándar es 0. Se requieren datos con mayor variabilidad.")

# --- TAB 5: EV & KELLY STAKE ---
with tab5:
    st.subheader("Calculadora de Valor Esperado (+EV) y Criterio de Kelly")
    
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    
    prob_estimada = col_ev1.number_input("Tu Probabilidad Estimada (%)", value=55.0, step=1.0) / 100
    cuota_casa = col_ev2.number_input("Cuota Ofrecida por la Casa de Apuestas", value=2.10, step=0.05)
    bankroll = col_ev3.number_input("Banca Total ($)", value=1000.0, step=100.0)
    
    ev = (prob_estimada * (cuota_casa - 1)) - (1 - prob_estimada)
    
    b = cuota_casa - 1
    p = prob_estimada
    q = 1 - p
    f_kelly = (b * p - q) / b if b > 0 else 0
    f_kelly_fractional = max(0.0, f_kelly * 0.25)
    
    stake_recomendado = f_kelly_fractional * bankroll
    
    st.divider()
    c_ev1, c_ev2, c_ev3 = st.columns(3)
    
    if ev > 0:
        c_ev1.metric("Expected Value (EV)", f"+{ev:.2%}", delta="¡Apuesta +EV con Valor!")
    else:
        c_ev1.metric("Expected Value (EV)", f"{ev:.2%}", delta="-EV (Sin Valor)", delta_color="inverse")
        
    c_ev2.metric("Kelly Stake Recomendado (1/4 Kelly)", f"${stake_recomendado:.2f}")
    c_ev3.metric("% del Bankroll a Arriesgar", f"{f_kelly_fractional*100:.2f}%")