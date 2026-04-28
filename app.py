import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="SPC 4.0 Dashboard", layout="wide", page_icon="📊")

# =========================
# PARAMÈTRES PROCESS
# =========================
LTS = 11.95
CIBLE = 12.00
LTI = 12.05
N = 5

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: #0E1117;
    color: white;
}

div[data-testid="stMetric"] {
    background: #1E1E1E;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #00D4FF55;
}

div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 30px;
    font-weight: 900;
}

div[data-testid="stMetricLabel"] p {
    color: #e5e7eb !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;color:#00D4FF;'>📊 SPC 4.0 - Contrôle Statistique Temps Réel</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>PFE 2026 | Diamètre Axe Moteur Ø12.00±0.05mm | IATF 16949 Inspired</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# GOOGLE SHEET
# =========================
SHEET_ID = "1vkfDof3og5G2YOizZP7WK-RCSx2IA2T75VgtMtV7fwM"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv(url)

    mesures_cols = ["Mesure1", "Mesure2", "Mesure3", "Mesure4", "Mesure5"]

    missing_cols = [col for col in mesures_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}")
        st.stop()

    if "Echantillon" not in df.columns:
        df["Echantillon"] = range(1, len(df) + 1)

    for col in mesures_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = df[df[mesures_cols].isna().any(axis=1)]

    if len(invalid_rows) > 0:
        st.error("❌ Certaines valeurs numériques sont invalides.")
        st.dataframe(invalid_rows, use_container_width=True)
        st.stop()

except Exception as e:
    st.error("🚨 Impossible de lire Google Sheet.")
    st.write(e)
    st.stop()

# =========================
# CALCULS SPC
# =========================
df["Xbar"] = df[mesures_cols].mean(axis=1)
df["R"] = df[mesures_cols].max(axis=1) - df[mesures_cols].min(axis=1)

A2 = 0.577
D3 = 0
D4 = 2.114
d2 = 2.326

Xbar_bar = df["Xbar"].mean()
R_bar = df["R"].mean()

LSC_X = Xbar_bar + A2 * R_bar
LIC_X = Xbar_bar - A2 * R_bar

LSC_R = D4 * R_bar
LIC_R = D3 * R_bar

sigma_est = R_bar / d2 if d2 > 0 else 0

if sigma_est > 0:
    Cp = (LTI - LTS) / (6 * sigma_est)
    Cpk = min(
        (LTI - Xbar_bar) / (3 * sigma_est),
        (Xbar_bar - LTS) / (3 * sigma_est)
    )
else:
    Cp = 0
    Cpk = 0

all_mesures = df[mesures_cols].values.flatten()
Pp = (LTI - LTS) / (6 * np.std(all_mesures)) if np.std(all_mesures) > 0 else 0

# =========================
# PDF FUNCTION
# =========================
def generate_pdf_report():
    doc_path = "rapport_qualite_specsense.pdf"
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Rapport Qualité - SPC 4.0", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Résumé global", styles["Heading2"]))
    story.append(Paragraph(f"Nombre d'échantillons : {len(df)}", styles["BodyText"]))
    story.append(Paragraph(f"Cible : {CIBLE:.2f} mm", styles["BodyText"]))
    story.append(Paragraph(f"LTS : {LTS:.2f} mm", styles["BodyText"]))
    story.append(Paragraph(f"LTI : {LTI:.2f} mm", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Indicateurs SPC", styles["Heading2"]))
    story.append(Paragraph(f"Moyenne Xbar : {Xbar_bar:.4f} mm", styles["BodyText"]))
    story.append(Paragraph(f"R moyen : {R_bar:.4f} mm", styles["BodyText"]))
    story.append(Paragraph(f"Sigma estimé : {sigma_est:.4f} mm", styles["BodyText"]))
    story.append(Paragraph(f"Cp : {Cp:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Cpk : {Cpk:.2f}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interprétation", styles["Heading2"]))

    if Cpk >= 1.33:
        conclusion = "Le processus est capable de respecter les tolérances client."
    elif Cpk >= 1.00:
        conclusion = "Le processus est limite. Une amélioration est nécessaire."
    else:
        conclusion = "Le processus n’est pas capable. Des actions correctives sont requises."

    story.append(Paragraph(conclusion, styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Actions recommandées", styles["Heading2"]))
    story.append(Paragraph("- Vérifier la stabilité du processus avec les cartes SPC.", styles["BodyText"]))
    story.append(Paragraph("- Analyser les causes racines en cas de point hors contrôle.", styles["BodyText"]))
    story.append(Paragraph("- Régler la machine pour recentrer le processus.", styles["BodyText"]))
    story.append(Paragraph("- Mettre en place un plan d’actions correctives.", styles["BodyText"]))

    doc = SimpleDocTemplate(doc_path)
    doc.build(story)

    return doc_path

# =========================
# KPI
# =========================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("📏 X̄ Moyenne", f"{Xbar_bar:.4f} mm", f"{(Xbar_bar - CIBLE) * 1000:.1f} µm")
col2.metric("📐 Cp", f"{Cp:.2f}", "Capable" if Cp >= 1.33 else "Non capable")
col3.metric("🎯 Cpk", f"{Cpk:.2f}", "Centré" if Cpk >= 1.33 else "Décentré")
col4.metric("📊 Sigma", f"{sigma_est:.4f} mm")
col5.metric("📈 Pp", f"{Pp:.2f}")

st.markdown("---")

# =========================
# CARTE XBAR
# =========================
fig_xbar = go.Figure()

fig_xbar.add_trace(go.Scatter(
    x=df["Echantillon"],
    y=df["Xbar"],
    mode="lines+markers",
    name="X̄"
))

fig_xbar.add_hline(y=LSC_X, line_dash="dash", line_color="red", annotation_text="LSC")
fig_xbar.add_hline(y=Xbar_bar, line_dash="solid", line_color="green", annotation_text="X̄̄")
fig_xbar.add_hline(y=LIC_X, line_dash="dash", line_color="red", annotation_text="LIC")
fig_xbar.add_hline(y=LTI, line_dash="dot", line_color="orange", annotation_text="LTI Spec")
fig_xbar.add_hline(y=LTS, line_dash="dot", line_color="orange", annotation_text="LTS Spec")
fig_xbar.add_hline(y=CIBLE, line_dash="solid", line_color="#00D4FF", annotation_text="Cible")

fig_xbar.update_layout(
    title="📈 Carte de contrôle X̄",
    template="plotly_dark",
    height=450,
    xaxis_title="Échantillon",
    yaxis_title="Diamètre [mm]"
)

st.plotly_chart(fig_xbar, use_container_width=True)

st.markdown("### 🧠 Interprétation X̄")
hors_controle_x = df[(df["Xbar"] > LSC_X) | (df["Xbar"] < LIC_X)]

if len(hors_controle_x) > 0:
    st.error(f"🔴 {len(hors_controle_x)} échantillon(s) hors limites de contrôle.")
else:
    st.success("✅ La carte X̄ montre un processus sous contrôle statistique.")

st.markdown("---")

# =========================
# CARTE R + HISTOGRAMME
# =========================
c1, c2 = st.columns(2)

with c1:
    fig_r = go.Figure()

    fig_r.add_trace(go.Scatter(
        x=df["Echantillon"],
        y=df["R"],
        mode="lines+markers",
        name="R"
    ))

    fig_r.add_hline(y=LSC_R, line_dash="dash", line_color="red", annotation_text="LSC")
    fig_r.add_hline(y=R_bar, line_dash="solid", line_color="green", annotation_text="R̄")
    fig_r.add_hline(y=LIC_R, line_dash="dash", line_color="red", annotation_text="LIC")

    fig_r.update_layout(
        title="📉 Carte R",
        template="plotly_dark",
        height=380,
        xaxis_title="Échantillon",
        yaxis_title="Étendue"
    )

    st.plotly_chart(fig_r, use_container_width=True)

with c2:
    fig_hist = px.histogram(
        x=all_mesures,
        nbins=25,
        title="📊 Distribution + Capabilité",
        template="plotly_dark"
    )

    fig_hist.add_vline(x=LTS, line_dash="dash", line_color="red", annotation_text="LTS")
    fig_hist.add_vline(x=LTI, line_dash="dash", line_color="red", annotation_text="LTI")
    fig_hist.add_vline(x=CIBLE, line_dash="solid", line_color="green", annotation_text="Cible")
    fig_hist.add_vline(x=Xbar_bar, line_dash="dot", line_color="orange", annotation_text="Moyenne")

    fig_hist.update_layout(
        height=380,
        xaxis_title="Diamètre [mm]",
        showlegend=False
    )

    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### 🧠 Interprétation Capabilité")

if Cpk >= 1.33:
    st.success("✅ Le processus est capable.")
elif Cpk >= 1.00:
    st.warning("🟡 Le processus est limite. Une amélioration est nécessaire.")
else:
    st.error("🔴 Le processus n’est pas capable. Action corrective requise.")

st.markdown("---")

# =========================
# ALERTES
# =========================
st.subheader("🚨 Alertes Qualité")

if len(hors_controle_x) > 0:
    st.error("🔴 ALERTE : point hors contrôle détecté.")
    st.write("Actions recommandées : arrêt ligne, isolation lot, analyse 5M, action corrective.")
else:
    st.success("✅ Aucun point hors contrôle détecté.")

if Cpk < 1.33:
    decentrage = (Xbar_bar - CIBLE) * 1000
    st.warning(f"⚠️ Capabilité insuffisante : Cpk = {Cpk:.2f}")
    st.write(f"Diagnostic : processus décentré de {decentrage:.1f} µm.")
    st.write(f"Action : régler la machine de {-decentrage:.1f} µm pour recentrer le processus.")
else:
    st.success("🏆 Processus capable selon le critère Cpk ≥ 1.33.")

# =========================
# TABLEAU
# =========================
st.markdown("---")
st.subheader("📋 Données brutes")

st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

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
