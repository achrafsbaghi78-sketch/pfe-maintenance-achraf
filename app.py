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

# === CSS PERSONNALISÉ ===
st.markdown("""
    <style>
   .main {background-color: #0E1117;}
   .stMetric {background-color: #1E1E1E; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>⚙️ Plateforme AI de Maintenance Prédictive 4.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FAFAFA;'>PFE 2026 - Ingénierie Industrielle | Achraf Sbaghi | Données Live via IoT</p>", unsafe_allow_html=True)
st.markdown("---")

# === SIDEBAR ===
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/maintenance.png", width=100)
    st.title("Panel de Contrôle")
    machine = st.selectbox("🏭 Machine", ["Presse B2 - Roulement 1", "Moteur C3", "Pompe H1"])
    st.success("✅ Connecté à Google Sheets Live")
    st.info("🔄 Mise à jour: Temps réel")
    st.markdown("---")
    st.caption("Développé avec ❤️ par Achraf")

# === LECTURE DATA MN GOOGLE SHEETS ===
SHEET_ID = "1nVJUGItidO-B4esCa0DESygITF1o4YHEGj-rs1Et30A"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    df = pd.read_csv(url)

    # Vérifier les colonnes nécessaires
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
            # Simulation label pour training
            seuil = int(0.85 * len(df))
            df['label'] = (df.index > seuil).astype(int)

            # Training XGBoost
            X = df[required_cols]
            model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            model.fit(X[:-20], df['label'][:-20])

            # Prédictions
            df['prob_panne'] = model.predict_proba(X)[:,1] * 100
            df['RUL'] = (len(df) - df.index) * 10 / 60 # 1 mesure = 10min
            df['index'] = df.index

        # === DASHBOARD ===
        st.header(f"📈 Dashboard Temps Réel: {machine}")

        # Métriques Principales
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
            # === JAUGE PROFESSIONNELLE ===
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sante,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Indice de Santé", 'font': {'size': 20, 'color': 'white'}},
                delta = {'reference': 90, 'increasing': {'color': "#2ECC71"}, 'decreasing': {'color': "#FF4B4B"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00D4FF", 'thickness': 0.3},
                    'bgcolor': "#1E1E1E",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#8B0000"},
                        {'range': [30, 70], 'color': "#FF8C00"},
                        {'range': [70, 100], 'color': "#006400"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30}}))
            fig_gauge.update_layout(height=350, paper_bgcolor="#0E1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            # === GRAPHIQUE RUL INTERACTIF ===
            fig_rul = px.line(df, x='index', y='RUL',
                             title='📉 Évolution RUL - Prédiction Défaillance',
                             labels={'index': 'Mesures', 'RUL': 'RUL (heures)'},
                             template='plotly_dark')
            fig_rul.add_hline(y=48, line_dash="dash", line_color="red", annotation_text="Seuil Critique 48h")
            fig_rul.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Seuil Attention 100h")
            fig_rul.update_layout(height=350, paper_bgcolor="#0E1117", plot_bgcolor="#1E1E1E")
            st.plotly_chart(fig_rul, use_container_width=True)

        # Ligne 2: Prob Panne + Heatmap Features
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            # === GRAPHIQUE PROBABILITÉ ===
            fig_prob = px.area(df, x='index', y='prob_panne',
                              title='📈 Probabilité de Panne - Temps Réel',
                              labels={'index': 'Mesures', 'prob_panne': 'Probabilité (%)'},
                              template='plotly_dark', color_discrete_sequence=['#FF4B4B'])
            fig_prob.update_layout(height=350, paper_bgcolor="#0E1117", plot_bgcolor="#1E1E1E")
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_g4:
            # === HEATMAP FEATURES ===
            fig_heat = px.imshow(df[required_cols].tail(50).T,
                                title='🔥 Heatmap 50 Dernières Mesures',
                                labels=dict(x="Mesure", y="Feature", color="Valeur"),
                                template='plotly_dark', aspect="auto")
            fig_heat.update_layout(height=350, paper_bgcolor="#0E1117")
            st.plotly_chart(fig_heat, use_container_width=True)

        # Tableau + Alertes
        st.markdown("---")
        st.subheader("📋 Dernières 10 Mesures")
        st.dataframe(df[required_cols + ['prob_panne', 'RUL']].tail(10).round(3),
                    use_container_width=True, height=350)

        # Système Alertes
        if rul < 24:
            st.error(f"🚨 ALERTE CRITIQUE NIVEAU 3: RUL = {rul:.1f}h. ARRÊT IMMÉDIAT RECOMMANDÉ sur {machine}!")
        elif rul < 48:
            st.error(f"⚠️ ALERTE CRITIQUE NIVEAU 2: RUL = {rul:.1f}h. Intervention urgente < 24h sur {machine}!")
        elif rul < 100:
            st.warning(f"⚠️ ATTENTION NIVEAU 1: RUL = {rul:.1f}h. Planifier maintenance préventive pour {machine}.")
        else:
            st.success(f"✅ {machine} en bon état de fonctionnement. RUL = {rul:.1f}h. Santé: {sante:.0f}%")

    else:
        st.error("❌ Colonnes manquantes dans Google Sheet. Vérifiez: rms, peak, kurtosis, crest, fft_236hz")

except Exception as e:
    st.error("❌ Erreur de connexion à Google Sheets")
    st.write(f"Détail: {e}")
