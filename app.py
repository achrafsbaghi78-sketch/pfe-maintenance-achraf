import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
st.set_page_config(page_title="SPC 4.0 Dashboard", layout="wide", page_icon="📊")

# === PARAMETRES PROCEDE - Diamètre Axe Moteur ===
LTS = 11.95 # Limite Tolérance Sup - Spec Client
CIBLE = 12.00 # Cible
LTI = 12.05 # Limite Tolérance Inf - Spec Client
N = 5 # Taille échantillon n=5

# === THEME ===
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.write("")
with col_btn:
    if st.button("🎨 Bdl Lown", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

if st.session_state.theme == 'light':
    bg_color, card_color, text_color, template = "#FFFFFF", "#F0F2F6", "#262730", "plotly_white"
    gauge_color = "#1f77b4"
else:
    bg_color, card_color, text_color, template = "#0E1117", "#1E1E1E", "#FAFAFA", "plotly_dark"
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
st.markdown(f"<h1 style='text-align: center; color: {gauge_color};'>📊 SPC 4.0 - Contrôle Statistique Temps Réel</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {text_color};'>PFE 2026 | Diamètre Axe Moteur Ø12.00±0.05mm | IATF 16949 Compliant</p>", unsafe_allow_html=True)
st.markdown("---")

# === LECTURE DATA - SHEET DYALK ===
SHEET_ID = "1vkfDof3og5G2YOizZP7WK-RCSx2IA2T75VgtMtV7fwM"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

try:
    df = pd.read_csv(url)
    mesures_cols = ['Mesure1','Mesure2','Mesure3','Mesure4','Mesure5']

    # Vérifier colonnes
    if not all(col in df.columns for col in mesures_cols):
        st.error("❌ Colonnes manquantes. Vérifiez: Mesure1, Mesure2, Mesure3, Mesure4, Mesure5")
        st.stop()

    # Calcul Xbar et R
    df['Xbar'] = df[mesures_cols].mean(axis=1)
    df['R'] = df[mesures_cols].max(axis=1) - df[mesures_cols].min(axis=1)

    # Constantes SPC pour n=5
    A2, D3, D4, d2 = 0.577, 0, 2.114, 2.326
    Xbar_bar = df['Xbar'].mean()
    R_bar = df['R'].mean()

    # Limites de contrôle
    LSC_X = Xbar_bar + A2 * R_bar
    LIC_X = Xbar_bar - A2 * R_bar
    LSC_R = D4 * R_bar
    LIC_R = D3 * R_bar

    # Capabilité
    sigma_est = R_bar / d2
    Cp = (LTI - LTS) / (6 * sigma_est)
    Cpk = min((LTI - Xbar_bar) / (3 * sigma_est), (Xbar_bar - LTS) / (3 * sigma_est))
    Pp = (LTI - LTS) / (6 * df[mesures_cols].values.std())

    # === KPIs ===
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📏 X̄̄ Moyenne", f"{Xbar_bar:.4f} mm", f"{(Xbar_bar-CIBLE)*1000:.1f} µm")
    col2.metric("📐 Cp", f"{Cp:.2f}", "Capable" if Cp>1.33 else "Non Capable", delta_color="off")
    col3.metric("🎯 Cpk", f"{Cpk:.2f}", "Centré" if Cpk>1.33 else "Décentré", delta_color="inverse")
    col4.metric("📊 Sigma σ", f"{sigma_est:.4f} mm")
    col5.metric("📈 Pp", f"{Pp:.2f}", "Perf." if Pp>1.67 else "Faible")

    st.markdown("---")

    # === CARTE XBAR ===
    fig_xbar = go.Figure()
    fig_xbar.add_trace(go.Scatter(x=df['Echantillon'], y=df['Xbar'], mode='lines+markers', name='X̄', line=dict(color=gauge_color, width=3)))
    fig_xbar.add_hline(y=LSC_X, line_dash="dash", line_color="#FF4B4B", annotation_text="LSC", annotation_position="right")
    fig_xbar.add_hline(y=Xbar_bar, line_dash="solid", line_color="#2ECC71", annotation_text="X̄̄", annotation_position="right")
    fig_xbar.add_hline(y=LIC_X, line_dash="dash", line_color="#FF4B4B", annotation_text="LIC", annotation_position="right")
    fig_xbar.add_hline(y=LTI, line_dash="dot", line_color="#FECB52", annotation_text="LTI Spec", annotation_position="left")
    fig_xbar.add_hline(y=LTS, line_dash="dot", line_color="#FECB52", annotation_text="LTS Spec", annotation_position="left")
    fig_xbar.add_hline(y=CIBLE, line_dash="solid", line_color="#00D4FF", annotation_text="Cible", annotation_position="left")
    fig_xbar.update_layout(title="📈 Carte de Contrôle X̄ - Moyennes Échantillons", template=template, height=450, paper_bgcolor=bg_color, plot_bgcolor=card_color, xaxis_title="N° Échantillon", yaxis_title="Diamètre [mm]")
    st.plotly_chart(fig_xbar, use_container_width=True)

    # === CARTE R + HISTOGRAMME ===
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df['Echantillon'], y=df['R'], mode='lines+markers', name='R', line=dict(color="#FF4B4B", width=3)))
        fig_r.add_hline(y=LSC_R, line_dash="dash", line_color="red", annotation_text="LSC")
        fig_r.add_hline(y=R_bar, line_dash="solid", line_color="green", annotation_text="R̄")
        fig_r.add_hline(y=LIC_R, line_dash="dash", line_color="red", annotation_text="LIC")
        fig_r.update_layout(title="📉 Carte R - Étendues", template=template, height=380, paper_bgcolor=bg_color, plot_bgcolor=card_color)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_g2:
        all_mesures = df[mesures_cols].values.flatten()
        fig_hist = px.histogram(x=all_mesures, nbins=25, title="📊 Distribution + Analyse Capabilité", template=template, color_discrete_sequence=[gauge_color])
        fig_hist.add_vline(x=LTS, line_dash="dash", line_color="#FF4B4B", annotation_text="LTS 11.95")
        fig_hist.add_vline(x=LTI, line_dash="dash", line_color="#FF4B4B", annotation_text="LTI 12.05")
        fig_hist.add_vline(x=CIBLE, line_dash="solid", line_color="#2ECC71", annotation_text="Cible 12.00")
        fig_hist.add_vline(x=Xbar_bar, line_dash="dot", line_color="#FECB52", annotation_text=f"X̄̄ {Xbar_bar:.4f}")
        fig_hist.update_layout(height=380, paper_bgcolor=bg_color, plot_bgcolor=card_color, xaxis_title="Diamètre [mm]", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    # === ALERTES AUTO IATF 16949 ===
    st.markdown("---")
    st.subheader("🚨 Système Alertes IATF 16949 - 8.5.1.1")

    hors_controle = df[(df['Xbar'] > LSC_X) | (df['Xbar'] < LIC_X)]
    regle_2 = False # 2/3 points en zone A
    regle_6 = False # 4/5 points du même côté

    if len(hors_controle) > 0:
        st.error(f"🔴 **RÈGLE 1 - ALERTE CRITIQUE**: {len(hors_controle)} échantillon(s) HORS LIMITES DE CONTRÔLE!")
        st.write(f"**Dernier point NOK**: Échantillon {hors_controle['Echantillon'].iloc[-1]} = {hors_controle['Xbar'].iloc[-1]:.4f} mm")
        st.write("**Action IATF 8.5.1.1 Requise:** 1. Arrêt ligne 2. Isolation lot 3. Analyse 5M 4. Action corrective 5. Validation")
    else:
        st.success("✅ **RÈGLE 1 OK** - Processus SOUS CONTRÔLE STATISTIQUE")

    if Cpk < 1.33:
        decentrage = (Xbar_bar - CIBLE) * 1000
        st.warning(f"⚠️ **CAPABILITÉ INSUFFISANTE**: Cpk = {Cpk:.2f} < 1.33")
        st.write(f"**Diagnostic**: Processus décentré de **{decentrage:.1f} µm** {'vers le haut' if decentrage>0 else 'vers le bas'}")
        st.write(f"**Action**: Régler machine de **{-decentrage:.1f} µm** pour centrer sur {CIBLE:.2f} mm")

    if Cp >= 1.33 and Cpk >= 1.33:
        st.success(f"🏆 **PROCESSUS CAPABLE** - Cp={Cp:.2f} | Cpk={Cpk:.2f} | Conforme IATF 16949")
if std_val > 0:
    cp = ...
    cpk = ...
else:
    cp = 0
    cpk = 0
    # === TABLEAU ===
    st.markdown("---")
    st.subheader("📋 Données Brutes - 10 Derniers
# =========================
# RAPPORT PDF
# =========================
st.markdown("---")
st.subheader("📄 Rapport Qualité")

if st.button("Générer le rapport PDF"):
    pdf_path = generate_pdf_report()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=f,
            file_name="rapport_qualite_specsense.pdf",
            mime="application/pdf"
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")
