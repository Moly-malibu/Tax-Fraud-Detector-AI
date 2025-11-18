# --------------------------------------------------------------
#  Multi-Fraud Detector AI + API + Batch + SHAP
#  Built by Liliana Bustamante | CPA Candidate | J.D. Law | 28 AI Certs
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import shap
import json
import base64
from io import BytesIO

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fraud Detector AI + API + SHAP",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Custom CSS ------------------
st.markdown(
    """
<style>
    .main {background:#fafafa;padding:2rem;border-radius:18px;box-shadow:0 6px 20px rgba(0,0,0,0.1);}
    .hero {background:linear-gradient(90deg,#1e3a8a,#3b82f6);padding:2.5rem;border-radius:18px;color:white;text-align:center;margin-bottom:2rem;box-shadow:0 10px 25px rgba(0,0,0,0.15);}
    .hero h1 {margin:0;font-size:3rem;font-weight:800;}
    .hero .highlight {color:#fbbf24;font-weight:700;}
    .shap-plot {border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#fff;}
    .footer {margin-top:4rem;font-size:0.95rem;color:#6b7280;text-align:center;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------ HERO ------------------
st.markdown(f"""
<div class="hero">
    <h1>Fraud Detector AI</h1>
    <p style="font-size:1.3rem;"><strong>Tax • Credit Card • Insurance</strong> + API + Batch + SHAP</p>
    <p style="font-size:1.15rem;">
        <span class="highlight">Real-time API</span> • <span class="highlight">CSV Batch</span> • <span class="highlight">Explainable AI</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# 1. DATA & MODELS (TAX, CC, INSURANCE)
# ==============================================================

@st.cache_data
def generate_irs_data():
    np.random.seed(42)
    n = 5000
    zipcode = np.random.randint(10000, 99999, n)
    num_returns = np.random.poisson(80, n) + 1
    income = np.random.lognormal(11, 0.8, n) * num_returns
    income = np.clip(income, 10000, 5_000_000)
    ded_ratio = np.random.uniform(0.05, 0.6, n)
    deductions = income * ded_ratio
    deductions = np.clip(deductions, 0, income * 0.8)
    taxable = income - deductions
    tax = taxable * np.random.uniform(0.15, 0.28, n)
    tax = np.clip(tax, 0, taxable * 0.4)

    df = pd.DataFrame({ "zipcode": zipcode, "num_returns": num_returns.astype(int), "income": income, "deductions": deductions, "tax": tax })
    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / (df["income"] + 1)
    df["is_fraud"] = (((df["ded_ratio"] > 0.5) | (df["tax_ratio"] < 0.05)) & (np.random.rand(n) < 0.35)).astype(int)
    return df

@st.cache_data
def generate_cc_data():
    np.random.seed(123)
    n = 8000
    amount = np.random.lognormal(4.2, 1.0, n); amount = np.clip(amount, 1, 10_000)
    time_since_last = np.random.exponential(300, n); time_since_last = np.clip(time_since_last, 0, 86_400)
    merchant_cat = np.random.choice(["grocery", "online", "travel", "fuel", "restaurant", "other"], n, p=[0.25,0.2,0.15,0.15,0.15,0.1])
    distance_km = np.random.gamma(2, 15, n); distance_km = np.clip(distance_km, 0, 500)

    df = pd.DataFrame({ "amount": amount, "time_since_last": time_since_last, "merchant_cat": merchant_cat, "distance_km": distance_km })
    df = pd.get_dummies(df, columns=["merchant_cat"], prefix="cat")
    df["amt_per_sec"] = df["amount"] / (df["time_since_last"] + 1)
    df["amt_per_km"] = df["amount"] / (df["distance_km"] + 1)
    fraud_cond = (df["amount"] > 3000) | (df["time_since_last"] < 60) | (df["distance_km"] > 200) | (df["amt_per_sec"] > 5)
    df["is_fraud"] = (fraud_cond & (np.random.rand(n) < 0.28)).astype(int)
    return df

@st.cache_data
def generate_insurance_data():
    np.random.seed(999)
    n = 6000
    age = np.random.randint(18, 85, n)
    vehicle_value = np.random.lognormal(9.5, 0.7, n); vehicle_value = np.clip(vehicle_value, 5000, 150000)
    claim_amount = np.random.lognormal(7.8, 1.1, n); claim_amount = np.clip(claim_amount, 100, 100000)
    days_since_policy = np.random.randint(30, 1825, n)
    prior_claims = np.random.poisson(0.4, n)
    injury_reported = np.random.choice([0,1], n, p=[0.7,0.3])
    police_report = np.random.choice([0,1], n, p=[0.6,0.4])
    witness = np.random.choice([0,1], n, p=[0.8,0.2])
    claim_type = np.random.choice(["collision", "comprehensive", "liability", "theft"], n, p=[0.4,0.3,0.2,0.1])

    df = pd.DataFrame({ "age": age, "vehicle_value": vehicle_value, "claim_amount": claim_amount, "days_since_policy": days_since_policy,
                     "prior_claims": prior_claims, "injury_reported": injury_reported, "police_report": police_report,
                     "witness": witness, "claim_type": claim_type })
    df = pd.get_dummies(df, columns=["claim_type"], prefix="type")
    df["claim_to_value"] = df["claim_amount"] / (df["vehicle_value"] + 1)
    df["high_claim"] = (df["claim_amount"] > 25000).astype(int)
    df["new_policy"] = (df["days_since_policy"] < 180).astype(int)
    fraud_cond = (df["claim_to_value"] > 0.9) | (df["claim_amount"] > 50000) | \
                 ((df["injury_reported"] == 1) & (df["police_report"] == 0)) | \
                 ((df["new_policy"] == 1) & (df["prior_claims"] == 0))
    df["is_fraud"] = (fraud_cond & (np.random.rand(n) < 0.32)).astype(int)
    return df

tax_df = generate_irs_data()
cc_df = generate_cc_data()
ins_df = generate_insurance_data()

@st.cache_resource
def train_models():
    # Tax
    X_tax = tax_df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]
    y_tax = tax_df["is_fraud"]
    tax_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    tax_model.fit(X_tax, y_tax)

    # CC
    cc_features = [c for c in cc_df.columns if c != "is_fraud"]
    X_cc = cc_df[cc_features]
    y_cc = cc_df["is_fraud"]
    cc_model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    cc_model.fit(X_cc, y_cc)

    # Insurance
    ins_features = [c for c in ins_df.columns if c != "is_fraud"]
    X_ins = ins_df[ins_features]
    y_ins = ins_df["is_fraud"]
    ins_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    ins_model.fit(X_ins, y_ins)

    return tax_model, cc_model, ins_model, cc_features, ins_features

tax_model, cc_model, ins_model, cc_features, ins_features = train_models()

# SHAP Explainers
@st.cache_resource
def get_shap_explainers():
    explainer_tax = shap.TreeExplainer(tax_model)
    explainer_cc = shap.TreeExplainer(cc_model)
    explainer_ins = shap.TreeExplainer(ins_model)
    return explainer_tax, explainer_cc, explainer_ins

explainer_tax, explainer_cc, explainer_ins = get_shap_explainers()

# ==============================================================
# 2. SIDEBAR: MODE + INPUTS
# ==============================================================

with st.sidebar:
    st.header("Fraud Module")
    fraud_type = st.radio("Select", ["Tax Return", "Credit Card", "Insurance Claim"])

    # === BATCH UPLOAD (TOP) ===
    st.markdown("---")
    st.subheader("Batch CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV for batch fraud scan", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} records")
            if st.button("Run Batch Fraud Scan", type="primary"):
                st.session_state.batch_run = True
                st.session_state.batch_df = batch_df
                st.session_state.batch_type = fraud_type.lower().replace(" ", "_")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Single Prediction")

    if fraud_type == "Tax Return":
        income = st.slider("Income ($)", 10000, 5_000_000, 800_000, step=10000)
        ded_pct = st.slider("Deduction %", 0, 80, 22)
        deductions = income * ded_pct / 100
        tax = st.number_input("Tax Paid ($)", 0, income, int((income - deductions) * 0.22))
        if st.button("Check Tax Fraud"):
            st.session_state.single_input = {"type": "tax", "income": income, "deductions": deductions, "tax": tax}
            st.session_state.run_single = True

    elif fraud_type == "Credit Card":
        amount = st.slider("Amount ($)", 1, 10000, 125)
        time_since_last = st.slider("Sec since last", 0, 86400, 300)
        merchant_cat = st.selectbox("Merchant", ["grocery", "online", "travel", "fuel", "restaurant", "other"])
        distance_km = st.slider("Distance (km)", 0, 500, 12)
        if st.button("Check CC Fraud"):
            st.session_state.single_input = {"type": "cc", "amount": amount, "time_since_last": time_since_last,
                              "merchant_cat": merchant_cat, "distance_km": distance_km}
            st.session_state.run_single = True

    else:  # Insurance
        age = st.slider("Age", 18, 85, 38)
        vehicle_value = st.slider("Vehicle ($)", 5000, 150000, 35000)
        claim_amount = st.slider("Claim ($)", 100, 100000, 8500)
        days_since_policy = st.slider("Days since policy", 30, 1825, 400)
        prior_claims = st.slider("Prior claims", 0, 10, 0)
        injury_reported = st.checkbox("Injury")
        police_report = st.checkbox("Police report", True)
        witness = st.checkbox("Witness")
        claim_type = st.selectbox("Type", ["collision", "comprehensive", "liability", "theft"])
        if st.button("Check Insurance Fraud"):
            st.session_state.single_input = {
                "type": "ins", "age": age, "vehicle_value": vehicle_value, "claim_amount": claim_amount,
                "days_since_policy": days_since_policy, "prior_claims": prior_claims,
                "injury_reported": injury_reported, "police_report": police_report,
                "witness": witness, "claim_type": claim_type
            }
            st.session_state.run_single = True

# ==============================================================
# 3. BATCH PROCESSING
# ==============================================================

if st.session_state.get("batch_run", False):
    df = st.session_state.batch_df.copy()
    typ = st.session_state.batch_type

    if typ == "tax_return":
        req_cols = ["income", "deductions", "tax"]
        model, features = tax_model, ["income", "deductions", "tax", "ded_ratio", "tax_ratio"]
        df["ded_ratio"] = df["deductions"] / df["income"]
        df["tax_ratio"] = df["tax"] / (df["income"] + 1)
    elif typ == "credit_card":
        req_cols = ["amount", "time_since_last", "merchant_cat", "distance_km"]
        model, features = cc_model, cc_features
        df = pd.get_dummies(df, columns=["merchant_cat"], prefix="cat")
        df["amt_per_sec"] = df["amount"] / (df["time_since_last"] + 1)
        df["amt_per_km"] = df["amount"] / (df["distance_km"] + 1)
    else:  # insurance
        req_cols = ["age", "vehicle_value", "claim_amount", "days_since_policy", "prior_claims",
                    "injury_reported", "police_report", "witness", "claim_type"]
        model, features = ins_model, ins_features
        df = pd.get_dummies(df, columns=["claim_type"], prefix="type")
        df["claim_to_value"] = df["claim_amount"] / (df["vehicle_value"] + 1)
        df["high_claim"] = (df["claim_amount"] > 25000).astype(int)
        df["new_policy"] = (df["days_since_policy"] < 180).astype(int)

    # Predict
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    df["fraud_probability"] = probs
    df["risk"] = np.where(probs > 0.7, "HIGH", np.where(probs > 0.3, "MEDIUM", "LOW"))

    st.success(f"Batch scan complete: {len(df)} records")
    st.dataframe(df[["fraud_probability", "risk"] + (req_cols[:3])].head(10))

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Full Report", csv, "batch_fraud_report.csv", "text/csv")

# ==============================================================
# 4. SINGLE PREDICTION + SHAP
# ==============================================================

if st.session_state.get("run_single", False):
    inp = st.session_state.single_input
    typ = inp["type"]

    if typ == "tax":
        ded_ratio = inp["deductions"] / inp["income"]
        tax_ratio = inp["tax"] / inp["income"]
        X = pd.DataFrame([{
            "income": inp["income"], "deductions": inp["deductions"], "tax": inp["tax"],
            "ded_ratio": ded_ratio, "tax_ratio": tax_ratio
        }])
        model, explainer = tax_model, explainer_tax
        feature_names = X.columns

    elif typ == "cc":
        amt_per_sec = inp["amount"] / (inp["time_since_last"] + 1)
        amt_per_km = inp["amount"] / (inp["distance_km"] + 1)
        X_dict = {c: [0.0] for c in cc_features}
        X_dict.update({
            "amount": [inp["amount"]], "time_since_last": [inp["time_since_last"]],
            "distance_km": [inp["distance_km"]], "amt_per_sec": [amt_per_sec], "amt_per_km": [amt_per_km],
            f"cat_{inp['merchant_cat']}": [1.0]
        })
        X = pd.DataFrame(X_dict)
        model, explainer = cc_model, explainer_cc
        feature_names = X.columns

    else:  # ins
        claim_to_value = inp["claim_amount"] / (inp["vehicle_value"] + 1)
        X_dict = {c: [0.0] for c in ins_features}
        X_dict.update({
            "age": [inp["age"]], "vehicle_value": [inp["vehicle_value"]], "claim_amount": [inp["claim_amount"]],
            "days_since_policy": [inp["days_since_policy"]], "prior_claims": [inp["prior_claims"]],
            "injury_reported": [int(inp["injury_reported"])], "police_report": [int(inp["police_report"])],
            "witness": [int(inp["witness"])], "claim_to_value": [claim_to_value],
            "high_claim": [int(inp["claim_amount"] > 25000)], "new_policy": [int(inp["days_since_policy"] < 180)],
            f"type_{inp['claim_type']}": [1.0]
        })
        X = pd.DataFrame(X_dict)
        model, explainer = ins_model, explainer_ins
        feature_names = X.columns

    prob = model.predict_proba(X)[0][1]
    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        title={'text': f"<b>{typ.upper()} RISK: {risk}</b>"},
        gauge={'bar': {'color': "#dc2626" if risk=="HIGH" else "#f59e0b" if risk=="MEDIUM" else "#10b981"}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Force Plot
    shap_values = explainer.shap_values(X)[1]  # class 1
    shap.force_plot(
        explainer.expected_value[1], shap_values[0], X.iloc[0], feature_names=feature_names,
        matplotlib=False, show=False, figsize=(12, 3)
    ).savefig("shap_plot.png", bbox_inches="tight", dpi=150)
    st.image("shap_plot.png", caption="Why this was flagged (SHAP)", use_column_width=True)

    # Alert
    if risk == "HIGH": st.error("**HIGH RISK** – Block & Investigate")
    elif risk == "MEDIUM": st.warning("**MEDIUM RISK** – Verify")
    else: st.success("**LOW RISK** – Safe")

# ==============================================================
# 5. API ENDPOINT (Copy to `api.py`)
# ==============================================================

api_code = '''
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Fraud Detector API")

# Load models (run once)
tax_model = joblib.load("tax_model.pkl")
cc_model = joblib.load("cc_model.pkl")
ins_model = joblib.load("ins_model.pkl")

class TaxInput(BaseModel):
    income: float
    deductions: float
    tax: float

class CCInput(BaseModel):
    amount: float
    time_since_last: float
    merchant_cat: str
    distance_km: float

class InsInput(BaseModel):
    age: int
    vehicle_value: float
    claim_amount: float
    days_since_policy: int
    prior_claims: int
    injury_reported: bool
    police_report: bool
    witness: bool
    claim_type: str

@app.post("/predict/tax")
def predict_tax(data: TaxInput):
    df = pd.DataFrame([data.dict()])
    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / df["income"]
    prob = tax_model.predict_proba(df)[0][1]
    return {"fraud_probability": prob, "risk": "HIGH" if prob>0.7 else "MEDIUM" if prob>0.3 else "LOW"}

# Add /predict/cc and /predict/insurance similarly...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

with st.expander("REAL-TIME API CODE (Copy to `api.py`)", expanded=False):
    st.code(api_code, language="python")

st.info("**To use API**: Save models with `joblib.dump(model, 'name.pkl')` and run `uvicorn api:app --reload`")

# ==============================================================
# 6. SAVE MODELS (RUN ONCE)
# ==============================================================

if st.button("Save Models for API (Run Once)"):
    import joblib
    joblib.dump(tax_model, "tax_model.pkl")
    joblib.dump(cc_model, "cc_model.pkl")
    joblib.dump(ins_model, "ins_model.pkl")
    st.success("Models saved: tax_model.pkl, cc_model.pkl, ins_model.pkl")

# ==============================================================
# 7. FOOTER
# ==============================================================

st.markdown(
    """
    <div class="footer">
        <strong>Built by Liliana Bustamante</strong> | CPA Candidate | J.D. Law | 28 AI Certifications<br>
        <a href="https://github.com/Moly-malibu/Tax-Fraud-Detector-AI" target="_blank">GitHub</a> •
        <a href="https://linkedin.com/in/your-profile" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)