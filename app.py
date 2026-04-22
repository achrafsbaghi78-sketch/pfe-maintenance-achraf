import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier
import numpy as np

# === CONFIGURATION PAGE ===
st.set_page_config(
    page_title="Smart Maintenance AI", 
    layout="wide", 
    page_icon="⚙️"
)

# === HEADER ===
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>⚙️ Plateforme AI de Maintenance Prédictive</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>PFE 2026 - Ingénierie Industrielle 4.0 | Achraf Sbaghi</p>", unsafe_allow_html=True)
st.markdown("---")

# === SIDEBAR ===
st.sidebar.image("https://img.icons8.com/color/96/000000/maintenance.png", width=80)
st.sidebar.title("Panel de Contrôle")
machine = st.sidebar.selectbox("🏭 Choisir Machine", ["Presse B2 - Roulement 1", "Moteur C3", "Pompe H1"])
st.sidebar.success("✅ Connecté à Google Sheets Live")
st.sidebar.info("🔄 Data mise à jour automatique")

# === LECTURE DATA MN GOOGLE SHEETS ===
SHEET_ID = "1nVJUGItidO-B4esCa0DESygITF1o4YHEGj-rs1Et30A"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    df = pd.read_csv(url)
    st.success("✅ Data Live mn Google Sheets chargée avec succès")
    
    # Vérifier les colonnes nécessaires
    required_cols = ['rms','peak','kurtosis','crest','fft_236hz']
    if all(col in df.columns for col in required_cols):
        
        with st.spinner('🤖 Calcul AI en cours...'):
            # Simulation label pour training - f production tji mn historique
            seuil = int(0.9 * len(df))
            df['label'] = (df.index > seuil).astype(int)
            
            # Training XGBoost
            X = df[required_cols]
            model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X[:-50], df['label'][:-50])
            
            # Prédictions
            df['prob_panne'] = model.predict_proba(X)[:,1] * 100
            df['RUL'] = (len(df) - df.index) * 10 / 60  # Simulation: 1 mesure = 10min
        
        # === DASHBOARD ===
        st.header(f"📈 Dashboard: {machine}")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        sante = 100 - df['prob_panne'].iloc[-1]
        rul = df['RUL'].iloc[-1]
        prob = df['prob_panne'].iloc[-1]
        
        col1.metric("Santé Machine", f"{sante:.0f}%")
        col2.metric("RUL Estimé", f"{rul:.1f} h")
        col3.metric("Prob. Panne", f"{prob:.1f}%")
        
        # Jauge Santé
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", 
            value = sante,
            title = {'text': "Indice de Santé Machine"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#1f77b4"},
                     'steps' : [
                         {'range': [0, 30], 'color': "#FF4B4B"},
                         {'range': [30, 70], 'color': "#FECB52"},
                         {'range': [70, 100], 'color': "#2ECC71"}],
                     'threshold': {
                         'line': {'color': "red", 'width': 4},
                         'thickness': 0.75,
                         'value': 30}}))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphiques RUL et Probabilité
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📉 Évolution RUL")
            st.line_chart(df[['RUL']])
        with col_g2:
            st.subheader("📈 Probabilité de Panne")
            st.line_chart(df[['prob_panne']])
        
        # Tableau dernières mesures
        st.subheader("📋 Dernières 10 Mesures")
        st.dataframe(df[required_cols + ['prob_panne', 'RUL']].tail(10).round(3), use_container_width=True)
        
        # Alertes
        st.markdown("---")
        if rul < 48:
            st.error(f"⚠️ ALERTE CRITIQUE: RUL = {rul:.1f}h seulement. Intervention urgente recommandée sur {machine}!")
        elif rul < 100:
            st.warning(f"⚠️ ATTENTION: RUL = {rul:.1f}h. Planifier maintenance préventive pour {machine}.")
        else:
            st.success(f"✅ {machine} en bon état. RUL = {rul:.1f}h. Prochaine inspection dans {(rul-100):.0f}h")
            
    else:
        st.error("❌ Colonnes manquantes dans Google Sheet. Vérifiez: rms, peak, kurtosis, crest, fft_236hz")
        st.write("Colonnes trouvées:", list(df.columns))

except Exception as e:
    st.error("❌ Erreur de connexion à Google Sheets")
    st.write("Vérifiez que:")
    st.write("1. Google Sheet partagé en 'Anyone with the link'")
    st.write("2. SHEET_ID correct")
    st.write("3. Colonnes: rms, peak, kurtosis, crest, fft_236hz")
    st.write(f"Détail erreur: {e}")
