# pages/_1_Tax_Fraud.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

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
    deductions = np.clip(income * ded_ratio, 0, income * 0.8)
    tax = np.clip((income - deductions) * np.random.uniform(0.15, 0.28, n), 0, None)

    df = pd.DataFrame({
        "zipcode": zipcode,
        "num_returns": num_returns.astype(int),
        "income": income,
        "deductions": deductions,
        "tax": tax
    })
    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / (df["income"] + 1)
    df["is_fraud"] = (((df["ded_ratio"] > 0.5) | (df["tax_ratio"] < 0.05)) & (np.random.rand(n) < 0.35)).astype(int)
    return df

df = generate_irs_data()

# ------------------ Train Model ------------------
@st.cache_resource
def train_model(_df):
    X = _df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]
    y = _df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    acc = (clf.predict(X_test) == y_test).mean()
    return clf, acc

model, acc = train_model(df)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Simulate a Return")
    zipcode = st.number_input("ZIP Code", 10000, 99999, 90210)
    num_returns = st.slider("Households", 1, 500, 50)
    income = st.slider("Total Income ($)", 10000, 5000000, 800000, step=10000)
    ded_pct = st.slider("Deduction %", 0, 80, 22)
    deductions = int(income * ded_pct / 100)
    tax = st.number_input("Tax Paid ($)", 0, income, int((income - deductions) * 0.22), step=1000)

    if st.button("Run Fraud Check", type="primary", use_container_width=True):
        st.session_state.tax_input = {
            "zipcode": zipcode, "num_returns": num_returns,
            "income": income, "deductions": deductions, "tax": tax
        }

# ------------------ Main Content ------------------
st.title("Tax Fraud Detector AI")
st.markdown("**AI-Powered IRS Audit Simulation** – Stop $450B in Fraud")

if "tax_input" in st.session_state:
    data = st.session_state.tax_input
    ded_ratio = data["deductions"] / data["income"]
    tax_ratio = data["tax"] / data["income"]

    input_row = pd.DataFrame({
        "income": [data["income"]], "deductions": [data["deductions"]], "tax": [data["tax"]],
        "ded_ratio": [ded_ratio], "tax_ratio": [tax_ratio]
    })

    prob = model.predict_proba(input_row)[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    color = "#dc2626" if risk == "HIGH" else "#f59e0b" if risk == "MEDIUM" else "#10b981"

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        title={'text': f"<b>Fraud Risk: {risk}</b> | Model Acc: {acc:.1%}"},
        gauge={'bar': {'color': color}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"${data['income']:,}")
    c2.metric("Deductions", f"${data['deductions']:,.0f}")
    c3.metric("Tax", f"${data['tax']:,}")
    c4.metric("Risk", f"{prob*100:.1f}%")

    # Alert
    if risk == "HIGH":
        st.error("**HIGH FRAUD RISK** – Recommend full audit.")
    elif risk == "MEDIUM":
        st.warning("**MODERATE RISK** – Review deduction sources.")
    else:
        st.success("**LOW RISK** – Return appears clean.")

    # Report
    report = f"""IRS Fraud Report
ZIP: {data['zipcode']} | Households: {data['num_returns']}
Income: ${data['income']:,} | Deductions: ${data['deductions']:,.0f} | Tax: ${data['tax']:,}
Fraud Probability: {prob*100:.1f}% | Risk: {risk}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    st.download_button("Download Report", report, "fraud_report.txt")

# ------------------ Dataset Insights ------------------
with st.expander("Dataset Insights & Fraud Patterns", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="income", nbins=50, title="Income Distribution", color_discrete_sequence=["#3b82f6"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sample = df.sample(1000)
        fig = px.scatter(sample, x="income", y="deductions", color="is_fraud",
                         color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                         hover_data=["zipcode"], title="Fraud Pattern (Red = High Risk)")
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Top ZIPs (NEW!) ------------------
st.markdown("### Top 10 High-Risk ZIP Codes")
top_zips = df[df["is_fraud"] == 1]["zipcode"].value_counts().head(10)
if len(top_zips) > 0:
    st.dataframe(pd.DataFrame({"ZIP": top_zips.index.astype(str), "Fraud Cases": top_zips.values}))
else:
    st.info("No fraud detected in sample.")