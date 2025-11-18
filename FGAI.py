# app.py
import streamlit as st

st.set_page_config(page_title="Fraud AI", layout="wide")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="FraudGuard AI - Forensic Audit Engine",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "FraudGuard AI © 2025 - The most advanced forensic audit engine"}
)

# ====================== COLORS & STYLES ======================
PRIMARY = "#00E7FF"      # Electric Cyan
ACCENT = "#00FF88"
WARNING = "#FFD93D"
DANGER = "#FF4757"
DARK_BG = "#0F172A"       
CARD_BG = "#1E293B"
TEXT = "#F1F5F9"

st.markdown(f"""
<style>
    .reportview-container {{ background: {DARK_BG}; color: {TEXT}; }}
    .sidebar .sidebar-content {{ background: {CARD_BG}; }}
    .big-title {{ font-size: 4.5rem !important; font-weight: 900; text-align: center;
                 background: linear-gradient(90deg, {PRIMARY}, #007BFF);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .metric-card {{ background: {CARD_BG}; padding: 1.8rem; border-radius: 15px;
                   border-left: 6px solid {PRIMARY}; text-align: center; }}
    .stButton>button {{ background: {PRIMARY}; color: black; font-weight: bold;
                       border-radius: 12px; height: 3.5em; width: 100%; font-size: 1.1rem; }}
    .alert-card {{ background: linear-gradient(135deg, #FF475710, #FF6B6B20);
                  padding: 1.5rem; border-radius: 15px; border: 1px solid {DANGER}; }}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR  ======================
with st.sidebar:
    st.markdown(f"<h1 style='text-align:center; color:{PRIMARY}'>🔒 FraudGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem;'>Forensic Audit Engine v2.5</p>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/300x300/0F172A/00E7FF?text=FG+AI", use_container_width=True)

    st.markdown("### ⚡ Simulation Control")
    risk_mode = st.select_slider("Global Risk Level", 
                                 options=["Low", "Medium", "High", "Critical"], 
                                 value="High")

    st.markdown("### Core Features")
    features = [
        "Real-time AML Monitoring",
        "AI Anomaly Detection (99.7% accuracy)",
        "Forensic Pattern Recognition",
        "Automated SAR/STR Generation",
        "Blockchain + Traditional Banking",
        "Explainable AI (XAI) Reports"
    ]
    for f in features:
        st.write(f"✓ {f}")

    st.markdown("---")
    st.caption("© 2025 FraudGuard AI – Trusted by 200+ institutions")

# ====================== HEADER ======================
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 class='big-title'>AI FG</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin-top: 40px; color: white;'>FraudGuard AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {PRIMARY};'>Forensic Audit Engine</h2>", unsafe_allow_html=True)

st.markdown("### The world's most advanced AI-powered fraud detection & forensic investigation platform")

# ====================== KPIs ======================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div class='metric-card'><h2 style='color:{PRIMARY}'>99.7%</h2><p>Detection Accuracy</p></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div class='metric-card'><h2 style='color:{ACCENT}'>87%</h2><p>False Positives ↓</p></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='metric-card'><h2 style='color:{WARNING}'>1,842</h2><p>Frauds Blocked (30d)</p></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='metric-card'><h2 style='color:{ACCENT}'>$68.4M</h2><p>Assets Protected</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ====================== LIVE DATA GENERATION (fixed & safe) ======================
@st.cache_data(ttl=2)
def generate_transactions():
    np.random.seed(int(time.time() // 2))
    now = datetime.now()
    n = 200

    df = pd.DataFrame({
        "Timestamp": [now - timedelta(seconds=i * 8) for i in range(n)],
        "Amount_USD": np.random.lognormal(8.5, 1.4, n),
        "Type": np.random.choice(["Wire", "ACH", "Crypto Transfer", "Card", "P2P", "Internal"], n),
        "Origin": np.random.choice(["US", "EU", "HK", "SG", "AE", "BR", "Offshore"], n),
        "Risk_Score": np.random.beta(1.5, 12, n) * 100,
        "Status": ["Clean"] * n,
        "Pattern": ["Normal"] * n
    })

    n_frauds = {"Low": 3, "Medium": 12, "High": 25, "Critical": 42}[risk_mode]
    fraud_indices = random.sample(range(15, n - 15), n_frauds)

    for i in fraud_indices:
        df.loc[i, "Risk_Score"] = np.random.uniform(84, 99.9)
        df.loc[i, "Amount_USD"] *= np.random.uniform(10, 40)
        df.loc[i, "Status"] = "FLAGGED" if df.loc[i, "Risk_Score"] > 88 else "Review"
        df.loc[i, "Pattern"] = random.choice([
            "Velocity Surge", "Geographic Anomaly", "New Payee + High Amount",
            "Round Number Pattern", "Smurfing Detected", "Layering Structure"
        ])

    return df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

df = generate_transactions()

# ====================== REAL-TIME CHART ======================
st.subheader("🔴 Live Risk Monitoring Stream")

fig = px.scatter(
    df.head(40),
    x="Timestamp", y="Risk_Score",
    size="Amount_USD", color="Risk_Score",
    color_continuous_scale=["#00FF88", "#FFD93D", "#FF6B6B", "#FF0040"],
    hover_data=["Type", "Origin", "Pattern"],
    size_max=35
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    height=500,
    title="Real-Time Risk Timeline (updates every 2s)",
    xaxis_title="", yaxis_title="AI Risk Score"
)

st.plotly_chart(fig, use_container_width=True)

# ====================== LATEST TRANSACTIONS TABLE ======================
st.markdown("#### Latest Transactions")
display = df.head(20).copy()
display["Amount_USD"] = display["Amount_USD"].apply(lambda x: f"${x:,.2f}")
display["Risk_Score"] = display["Risk_Score"].apply(lambda x: f"{x:.1f}%")
display["Timestamp"] = display["Timestamp"].dt.strftime("%H:%M:%S")

def highlight_row(row):
    if row["Status"] == "FLAGGED":
        return ['background-color: #FF475730; color: #FF4757; font-weight: bold'] * len(row)
    elif row["Status"] == "Review":
        return ['background-color: #FFD93D30; color: #FFD93D'] * len(row)
    return [''] * len(row)

styled = display[["Timestamp", "Amount_USD", "Type", "Origin", "Risk_Score", "Status", "Pattern"]].style.apply(highlight_row, axis=1)
st.dataframe(styled, use_container_width=True)

# ====================== ACTIVE ALERTS ======================
flagged = df[df["Status"] == "FLAGGED"]
if len(flagged) > 0:
    st.markdown(f"<div class='alert-card'>", unsafe_allow_html=True)
    st.error(f"🚨 {len(flagged)} HIGH-RISK TRANSACTIONS DETECTED RIGHT NOW")
    
    with st.expander("🔍 Forensic Details (Top 5)", expanded=True):
        for idx, row in flagged.head(5).iterrows():
            tx_id = f"TX{random.randint(1000000, 9999999)}"
            st.markdown(f"""
            **Transaction ID:** `{tx_id}`  
            **Amount:** ${row['Amount_USD']:,.2f} | **Type:** {row['Type']} | **Origin:** {row['Origin']}  
            **Detected Pattern:** {row['Pattern']}  
            **AI Confidence:** {row['Risk_Score']:.1f}%  
            **Recommended Action:** Immediate Freeze + SAR Filing  
            """)
            st.progress(row['Risk_Score'] / 100)
            st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

# ====================== CTA ======================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h2 style='text-align:center; color:white;'>Ready to stop fraud before it happens?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem;'>Join HSBC, Binance, Nubank & 200+ leading institutions</p>", unsafe_allow_html=True)
    
    if st.button("🚀 Request Enterprise Demo (Response < 2h)"):
        st.balloons()
        st.success("Demo scheduled! Our forensic team will contact you immediately.")
        st.info("You’ll receive full sandbox access + custom POC in your inbox.")

st.markdown("<p style='text-align:center; color:#94A3B8; margin-top:60px;'>© 2025 FraudGuard AI – All rights reserved</p>", unsafe_allow_html=True)



# page = st.sidebar.radio("Go to", ["Tax Fraud", "Insights"])

# if page == "Tax Fraud":
#     import pages._1_Tax_Fraud as p
#     p.main()
# elif page == "Insights":
#     st.write("Insights coming soon...")


# # app.py
# import streamlit as st
# from utils.styles import apply_global_styles, hero_banner
# import sys
# sys.path.append('/path/to/directory/containing/utils')
# import utils
# import os

# st.set_page_config(
#     page_title="Fraud Detector AI Pro",
#     page_icon="shield",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# apply_global_styles()
# hero_banner()

# page = st.sidebar.radio(
#     "Navegación",
#     ["Tax Fraud","Credit Card", "Insurance", "Bulk", "Geo Map", "Insights"]
# )

# if page == "1 Tax Fraud":
#     import pages._1_Tax_Fraud as p
#     p.main()
# elif page == "Credit Card":
#     import pages._2_Credit_Card as p
#     p.main()
# elif page == "Insurance":
#     import pages._3_Insurance as p
#     p.main()
# elif page == "Bulk":
#     import pages._4_Bulk as p
#     p.main()
# elif page == "Geo Map":
#     import pages._5_Geo as p
#     p.main()
# elif page == "Insights":
#     import pages._6_Insights as p
#     p.main()