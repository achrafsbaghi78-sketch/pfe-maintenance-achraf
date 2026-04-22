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
    initial_sidebar_state="collapsed"
)

# === INITIALISER THEME STATE ===
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# === BUTTON DYAL THEME LFO9 ===
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.write("")
with col_btn:
    if st.button("🎨 Bdl Lown", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# === CSS DYNAMIQUE ===
if st.session_state.theme == 'light':
    bg_color = "#FFFFFF"
    card_color = "#F0F2F6"
    text_color = "#262730"
    plotly_template = "plotly_white"
    gauge_color = "#1f77b4"
else:
    bg_color = "#0E1117"
    card_color = "#1E1E1E"
    text_color = "#FAFAFA"
    plotly_template = "plotly_dark"
    gauge_color = "#00D4FF"

st.markdown(f"""
    <style>
.main {{background-color: {bg_color};}}
.stMetric {{background-color: {card_color}; padding: 15px; border-radius: 10px; border: 1px solid {gauge_color}40;}}
   h1, h2, h3, p, label {{color: {text_color}!important;}}
.stButton>button {{background-color: {card_color}; color: {text_color}; border: 1px solid {gauge_color};}}
    </style>
    """, unsafe_allow_html=True)

# === HEADER ===
st.markdown(f"<h1 style='text-align: center; color: {gauge_color};'>⚙️ Plateforme AI de Maintenance Prédictive 4.0</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {text_color};'>PFE 2026 - Ingénierie Industrielle | Données Live via IoT</p>", unsafe_allow_html=True)
st.markdown("---")

# === SIDEBAR ===
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/maintenance.png", width=100)
    st.title("Panel Info")
    machine = st.selectbox("🏭 Machine", ["Presse B2 - Roulement 1", "Moteur C3", "Pompe H1"])
    st.success("✅ Connecté à Google Sheets Live")
    st.info("🔄 Mise à jour: Temps réel")
    st.markdown("---")
    st.caption("Développé avec ❤️ par Achraf Sbaghi")

# === LECTURE DATA ===
SHEET_ID = "1nVJUGItidO-B4es
