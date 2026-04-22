import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
import numpy as np

# === CONFIGURATION PAGE ===
st.set_page_config(
    page_title="Smart Maintenance AI",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# === SIDEBAR - CONTROLES ===
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/maintenance.png", width=100)
    st.title("Panel de Contrôle")

    # === TOGGLE LIGHT/DARK MODE 🌗 ===
    dark_mode = st.toggle("🌙 Dark Mode", value=True, help="Bdel bin Dark w Light Mode")

    st.markdown("---")
    machine = st.selectbox("🏭 Machine", ["Presse B2 - Roulement 1", "Moteur C3", "Pompe H1"])
    st.success("✅ Connecté à Google Sheets Live")
    st.info("🔄 Mise à jour: Temps réel")
    st.markdown("---")
    st.caption("Développé avec ❤️ par Achraf Sbaghi")

# === CSS DYNAMIQUE SELON THEME ===
if dark_mode:
    bg_color = "#0E1117"
    card_color = "#1E1E1E"
    text_color = "#FAFAFA"
    plotly_template = "plotly_dark"
    gauge_color = "#00D4FF"
else:
    bg_color = "#FFFFFF"
    card_color = "#F0F2F6"
    text_color = "#262730"
    plotly_template = "plotly_white"
    gauge_color = "#1f77b4"

st.markdown(f"""
    <style>
 .main {{background-color: {bg_color};}}
 .stMetric {{background-color: {card_color}; padding: 15px; border-radius: 10px;}}
   h1, h2, h3, p {{color: {text_color}!important;}}
    </style>
    """, unsafe_allow_html=True)

# === HEADER ===
st.markdown(f"<h1 style='text-align: center; color: {gauge_color};'>⚙️ Plateforme AI de Maintenance Prédictive 4.0</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {text_color};'>PFE 2026 - Ingénierie Industrielle | Données Live via IoT</p>", unsafe_allow_html=True)
st.markdown("---")

# === LECTURE DATA MN GOOGLE SHEETS ===
SHEET_ID = "1nVJUGItidO-B4esCa0DESygITF1o4YHEGj-rs1Et30A"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    df = pd.read_csv(url)
    required_cols = ['rms','peak','kurtosis','crest','fft_236hz']

    if all(col in df.columns for col in required_cols):
        # FIX: CONVERTI L AR9AM
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)

        if len(df) < 20:
            st.warning(f"⚠️ 3ndek ghir {len(df)} mesure. Khass 20+ bach dashboard yban mzian")
            st.stop()

        with st.spinner('🤖 Analyse AI XGBoost en cours...'):
            seuil = int(0.85 * len(df))
            df['label'] = (df.index > seuil).astype(int)
            X = df[required_cols]
            model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            model.fit(X[:-20], df['label'][:-20])
            df['prob_panne'] = model.predict_proba(X)[:,1] * 100
            df['RUL'] = (len(df) - df.index) * 10 / 60
            df['index'] = df.index

        # === DASHBOARD ===
        st.header(f"📈 Dashboard Temps Réel: {machine}")

        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        sante = 100 - df['prob_panne'].iloc[-1]
        rul = df['RUL'].iloc[-1]
        prob = df['prob_panne'].iloc[-1]
        rms_actuel = df['rms'].iloc[-1]

        col1.metric("💚 Santé Machine", f"{sante:.1f}%", f"{sante-90:.1f}%" if sante<90 else "Stable")
        col2.metric("⏱️ RUL Estimé", f"{rul:.1f} h", f"-{df['RUL'].iloc[-2]-rul:.1f}h", delta_color="inverse")
        col3.metric("🚨 Prob. Panne", f"{prob:.1f}%", f"+{prob-df['prob_panne'].iloc[-2]:.1f}%", delta_color="inverse")
        col4.metric("📊 RMS Actuel", f"{rms_actuel:.3f}", f"{(rms_actuel-df['rms'].iloc[-2])/df['rms'].iloc[-2]*100:.1f}%")

        st.markdown("---")

        # Ligne 1: Jauge + RUL Graph
        col_g1, col_g2 = st.columns([1,2])

        with col_g1:
            # === JAUGE CORRIGÉE ===
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sante,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Indice de Santé", 'font': {'size': 20, 'color': text_color}},
                delta = {'reference': 90},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': text_color},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': card_color,
                    'steps': [
                        {'range': [0, 30], 'color': "#FF4B4B"},
                        {'range': [30, 70], 'color': "#FECB52"},
                        {'range': [70, 100], 'color': "#2ECC71"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 30}
                }
            ))
            fig_gauge.update_layout(height=350, paper_bgcolor=bg_color, font={'color': text_color})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            fig_rul = px.line(df, x='index', y='RUL', title='📉 Évolution RUL', template=plotly_template)
            fig_rul.add_hline(y=48, line_dash="dash", line_color="red", annotation_text="Seuil Critique 48h")
            fig_rul.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Seuil Attention 100h")
            fig_rul.update_layout(height=350, paper_bgcolor=bg_color, plot_bgcolor=card_color)
            st.plotly_chart(fig_rul, use_container_width=True)

        # Ligne 2: Prob + Heatmap
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            fig_prob = px.area(df, x='index', y='prob_panne', title='📈 Probabilité de Panne',
                              template=plotly_template, color_discrete_sequence=['#FF4B4B'])
            fig_prob.update_layout(height=350, paper_bgcolor=bg_color, plot_bgcolor=card_color)
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_g4:
            fig_heat = px.imshow(df[required_cols].tail(50).T, title='🔥 Heatmap 50 Dernières Mesures',
                                template=plotly_template, aspect="auto")
            fig_heat.update_layout(height=350, paper_bgcolor=bg_color)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Dernières 10 Mesures")
        st.dataframe(df[required_cols + ['prob_panne', 'RUL']].tail(10).round(3), use_container_width=True, height=350)

        # Alertes
        if rul < 24:
            st.error(f"🚨 ALERTE CRITIQUE N3: RUL = {rul:.1f}h. ARRÊT IMMÉDIAT RECOMMANDÉ sur {machine}!")
        elif rul < 48:
            st.error(f"⚠️ ALERTE CRITIQUE N2: RUL = {rul:.1f}h. Intervention < 24h sur {machine}!")
        elif rul < 100:
            st.warning(f"⚠️ ATTENTION N1: RUL = {rul:.1f}h. Planifier maintenance pour {machine}.")
        else:
            st.success(f"✅ {machine} en bon état. RUL = {rul:.1f}h. Santé: {sante:.0f}%")

    else:
        st.error("❌ Colonnes manquantes. Vérifiez: rms, peak, kurtosis, crest, fft_236hz")

except Exception as e:
    st.error(f"❌ Erreur: {e}")
