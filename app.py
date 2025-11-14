# --------------------------------------------------------------
#  Tax Fraud Detector AI – Simulated IRS 2021 Data (NO ERRORS)
#  Deploy: https://tax-fraud-detector.streamlit.app
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Tax Fraud Detector AI",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ CSS ------------------
st.markdown(
    """
<style>
    .main {background:#fafafa;padding:2rem;border-radius:18px;box-shadow:0 6px 20px rgba(0,0,0,0.1);}
    .title {font-size:3rem;font-weight:800;color:#1d4ed8;text-align:center;}
    .subtitle {font-size:1.3rem;color:#475569;text-align:center;margin-bottom:2rem;}
    .metric-box {background:#e0e7ff;padding:1rem;border-radius:12px;text-align:center;font-weight:600;}
    .info-box {background:#dbeafe;padding:1rem;border-radius:8px;border-left:4px solid #3b82f6;}
    .footer {margin-top:4rem;font-size:0.95rem;color:#6b7280;text-align:center;}
</style>
""",
    unsafe_allow_html=True,  # CORRECT
)

# ------------------ Header ------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">Tax Fraud Detector AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Simulated IRS 2021 SOI Data – Realistic fraud detection with ML</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-box">
<strong>Built by Liliana Bustamante</strong> | CPA Candidate | 28 AI Certs | Lawyer<br>
        Data: Simulated from IRS 2021 SOI | <a href="https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-data" target="_blank">IRS.gov SOI Stats</a><br>
        <a href="https://github.com/your-username/tax-fraud-detector" target="_blank">GitHub</a> •
        <a href="https://linkedin.com/in/your-profile" target="_blank">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------ Simulated IRS Data ------------------
@st.cache_data
def generate_irs_data():
    np.random.seed(42)
    n = 5000
    zipcode = np.random.randint(10000, 99999, n)
    num_returns = np.random.poisson(80, n) + 1
    income = np.random.lognormal(11, 0.8, n) * num_returns
    income = np.clip(income, 10000, 5000000)
    ded_ratio = np.random.uniform(0.05, 0.6, n)
    deductions = income * ded_ratio
    deductions = np.clip(deductions, 0, income * 0.8)
    taxable = income - deductions
    tax = taxable * np.random.uniform(0.15, 0.28, n)
    tax = np.clip(tax, 0, taxable * 0.4)

    df = pd.DataFrame({
        "zipcode": zipcode,
        "num_returns": num_returns.astype(int),
        "income": income,
        "deductions": deductions,
        "tax": tax
    })

    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / (df["income"] + 1)
    df["is_fraud"] = (
        ((df["ded_ratio"] > 0.5) | (df["tax_ratio"] < 0.05))
        & (np.random.rand(n) < 0.35)
    ).astype(int)

    return df

full_df = generate_irs_data()

# ------------------ Train Model ------------------
@st.cache_resource
def train_model(_df):
    X = _df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]
    y = _df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    acc = (clf.predict(X_test) == y_test).mean()
    return clf, acc

model, model_acc = train_model(full_df)

# ------------------ Sidebar (ALL INT) ------------------
with st.sidebar:
    st.header("Simulate a Return")
    zipcode = st.number_input("ZIP Code", 10000, 99999, 90210, step=1)
    num_returns = st.slider("Households", 1, 500, 50, step=1)
    income = st.slider("Total Income ($)", 10000, 5000000, 800000, step=10000)
    ded_pct = st.slider("Deduction %", 0, 80, 22, step=1)
    deductions = int(income * ded_pct / 100)
    tax = st.number_input("Tax Paid ($)", 0, income, int((income - deductions) * 0.22), step=1000)

    if st.button("Run Fraud Check", type="primary", use_container_width=True):
        st.session_state.run = True
    else:
        st.session_state.run = False

# ------------------ Prediction ------------------
if st.session_state.get("run", False):
    input_row = pd.DataFrame({
        "income": [float(income)],
        "deductions": [float(deductions)],
        "tax": [float(tax)],
        "ded_ratio": [deductions / income],
        "tax_ratio": [tax / income]
    })

    prob = model.predict_proba(input_row)[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    color = {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#10b981"}[risk]

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': f"<b>Risk: {risk}</b> – Acc: {model_acc:.1%}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fecaca'}
            ],
            'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 70}
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1, c2, c3, c4],
                               ["Income", "Deductions", "Tax", "Risk"],
                               [f"${income:,}", f"${deductions:,}", f"${tax:,}", f"{prob*100:.1f}%"]):
        with col:
            st.markdown(f"<div class='metric-box'>{label}<br><b>{val}</b></div>", unsafe_allow_html=True)

    # Alert
    if risk == "HIGH":
        st.error("High fraud risk – matches IRS audit patterns.")
    elif risk == "MEDIUM":
        st.warning("Moderate risk – review deduction sources.")
    else:
        st.success("Low risk – return appears clean.")

    # Download report
    report = f"""IRS Fraud Report
ZIP: {zipcode} | Households: {num_returns}
Income: ${income:,} | Deductions: ${deductions:,} | Tax: ${tax:,}
Fraud Probability: {prob*100:.1f}% | Risk: {risk}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    st.download_button("Download Report", report, "fraud_report.txt", "text/plain")

# ------------------ Data Explorer ------------------
with st.expander("Explore Simulated IRS Dataset", expanded=False):
    st.dataframe(full_df.head(10).style.format({
        'income': '${:,.0f}', 'deductions': '${:,.0f}', 'tax': '${:,.0f}'
    }))
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(values=full_df['is_fraud'].value_counts().values,
                         names=["Clean", "Potential Fraud"],
                         color_discrete_sequence=["#86efac", "#fca5a5"])
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        sample = full_df.sample(800)
        fig_scatter = px.scatter(sample, x="income", y="deductions", color="is_fraud",
                                 color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                                 hover_data=["zipcode"])
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------ Footer ------------------
st.markdown(
    """
    <div class="footer">
        <strong>Built by Liliana Bustamante</strong> | CPA Candidate | Data Science and Machine Learning AI Certs | J.D. Lawyer<br>
        Data: Simulated from IRS 2021 SOI | <a href="https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income" target="_blank">IRS.gov</a><br>
        <a href="https://github.com/Moly-malibu/Tax-Fraud-Detector-AI.git" target="_blank">GitHub</a> •
        <a href="https://linkedin.com/in/your-profile" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)