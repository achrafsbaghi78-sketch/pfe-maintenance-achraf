import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier
import numpy as np

st.set_page_config(page_title="Smart Maintenance AI", layout="wide", page_icon="⚙️")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>⚙️ Plateforme AI de Maintenance Prédictive</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>PFE 2026 - Ingénierie Industrielle 4.0 | Achraf Sbaghi</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.image("https://img.icons8.com/color/96/000000/maintenance.png", width=80)
st.sidebar.title("Panel de Contrôle")
machine = st.sidebar.selectbox("🏭 Choisir Machine", ["Presse B2 - Roulement 1", "Moteur C3", "Pompe H1"])
st.sidebar.success("✅ Déployé sur Streamlit Cloud")

# === KAY9RA DATA MN GOOGLE SHEETS BO7DO ===
SHEET_ID = "1ABC_XYZ_123"  # GADI TBDEL HADI B ID DYALK
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    df = pd.read_csv(url)
    st.success("✅ Data Live mn Google Sheets")
    uploaded_file = True  # Bach code l9dim ykhdem
except:
    st.error("❌ Vérifier ID dyal Google Sheet w Public access")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['rms','peak','kurtosis','crest','fft_236hz']
    if all(col in df.columns for col in required_cols):
        with st.spinner('🤖 Calcul AI en cours...'):
            seuil = int(0.9 * len(df))
            df['label'] = (df.index > seuil).astype(int)
            X = df[required_cols]
            model = XGBClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X[:-50], df['label'][:-50])
            df['prob_panne'] = model.predict_proba(X)[:,1] * 100
            df['RUL'] = (len(df) - df.index) * 10 / 60
        
        st.header(f"📈 Dashboard: {machine}")
        col1, col2, col3 = st.columns(3)
        sante = 100 - df['prob_panne'].iloc[-1]
        rul = df['RUL'].iloc[-1]
        prob = df['prob_panne'].iloc[-1]
        
        col1.metric("Santé Machine", f"{sante:.0f}%")
        col2.metric("RUL Estimé", f"{rul:.1f} h")
        col3.metric("Prob. Panne", f"{prob:.1f}%")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = sante,
            title = {'text': "Indice de Santé"},
            gauge = {'axis': {'range': [0, 100]},
                     'steps' : [{'range': [0, 30], 'color': "#FF4B4B"},
                                 {'range': [30, 70], 'color': "#FECB52"},
                                 {'range': [70, 100], 'color': "#2ECC71"}]}))
        st.plotly_chart(fig, use_container_width=True)
        
        st.line_chart(df[['RUL', 'prob_panne']])
        
        if rul < 48:
            st.error(f"⚠️ ALERTE CRITIQUE: RUL = {rul:.1f}h. Intervention urgente!")
        elif rul < 100:
            st.warning(f"⚠️ ATTENTION: RUL = {rul:.1f}h. Planifier maintenance.")
        else:
            st.success("✅ Machine en bon état")
    else:
        st.error("❌ Colonnes manquantes")
else:
    st.info("👆 Uploadi fichier features_complet.csv dyalk mn Colab")
