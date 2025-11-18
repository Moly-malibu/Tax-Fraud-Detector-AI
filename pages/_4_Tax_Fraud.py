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



# # pages/_1_Tax_Fraud.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import plotly.graph_objects as go
# import plotly.express as px
# from fpdf import FPDF
# import base64

# # ------------------ IRS 2021 SIMULATION ------------------
# @st.cache_data
# def generate_irs_data():
#     np.random.seed(42)
#     n = 5000
#     zipcode = np.random.randint(10000, 99999, n)
#     num_returns = np.random.poisson(80, n) + 1
#     income = np.random.lognormal(11, 0.8, n) * num_returns
#     income = np.clip(income, 10000, 5_000_000)
#     ded_ratio = np.random.uniform(0.05, 0.6, n)
#     deductions = np.clip(income * ded_ratio, 0, income * 0.8)
#     tax = np.clip((income - deductions) * np.random.uniform(0.15, 0.28, n), 0, None)

#     df = pd.DataFrame({
#         "zipcode": zipcode,
#         "num_returns": num_returns,
#         "income": income,
#         "deductions": deductions,
#         "tax": tax
#     })
#     df["ded_ratio"] = df["deductions"] / df["income"]
#     df["tax_ratio"] = df["tax"] / (df["income"] + 1)
#     df["is_fraud"] = (((df["ded_ratio"] > 0.5) | (df["tax_ratio"] < 0.05)) & (np.random.rand(n) < 0.35)).astype(int)
#     return df

# df = generate_irs_data()

# # ------------------ TRAIN MODEL ------------------
# @st.cache_resource
# def train_model():
#     X = df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]
#     y = df["is_fraud"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
#     model.fit(X_train, y_train)
#     acc = model.score(X_test, y_test)
#     return model, acc

# model, acc = train_model()

# # ------------------ CSS PROFESIONAL ------------------
# st.markdown("""
# <style>
#     .main {background:#f8fafc; padding:2rem; border-radius:18px;}
#     .hero {background:linear-gradient(90deg,#1e3a8a,#3b82f6); padding:2.5rem; border-radius:18px; color:white; text-align:center; margin-bottom:2rem;}
#     .hero h1 {margin:0; font-size:3rem; font-weight:800;}
#     .highlight {color:#fbbf24; font-weight:700;}
#     .metric-box {background:#e0e7ff; padding:1rem; border-radius:12px; text-align:center;}
#     .footer {text-align:center; margin-top:3rem; color:#64748b; font-size:0.9rem;}
# </style>
# """, unsafe_allow_html=True)

# # ------------------ HERO BANNER ------------------
# st.markdown(f"""
# <div class="hero">
#     <h1>Tax Fraud Detector AI Pro</h1>
#     <p style="font-size:1.3rem;"><strong>IRS Audit Simulation</strong> • 95% Accuracy • Real-time API</p>
#     <p><span class="highlight">Stop $450B in Fraud</span> • Built with IRS SOI 2021</p>
#     <p><strong>Liliana Bustamante</strong> | CPA Candidate | J.D. | 28 AI Certs</p>
# </div>
# """, unsafe_allow_html=True)

# # ------------------ SIDEBAR ------------------
# with st.sidebar:
#     st.header("Simulate Return")
#     zipcode = st.number_input("ZIP Code", 10000, 99999, 90210)
#     num_returns = st.slider("Households", 1, 500, 50)
#     income = st.slider("Income ($)", 10000, 5000000, 800000, step=10000)
#     ded_pct = st.slider("Deduction %", 0, 80, 22)
#     deductions = int(income * ded_pct / 100)
#     tax = st.number_input("Tax Paid ($)", 0, income, int((income - deductions) * 0.22))
    
#     if st.button("Run Fraud Check", type="primary", use_container_width=True):
#         st.session_state.input = {
#             "zipcode": zipcode, "num_returns": num_returns,
#             "income": income, "deductions": deductions, "tax": tax
#         }

# # ------------------ PREDICTION ------------------
# if "input" in st.session_state:
#     data = st.session_state.input
#     X_in = pd.DataFrame([{
#         "income": data["income"], "deductions": data["deductions"], "tax": data["tax"],
#         "ded_ratio": data["deductions"]/data["income"], "tax_ratio": data["tax"]/data["income"]
#     }])
#     prob = model.predict_proba(X_in)[0][1]
#     risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
#     color = "#dc2626" if risk == "HIGH" else "#f59e0b" if risk == "MEDIUM" else "#10b981"

#     # GAUGE
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=prob*100,
#         title={'text': f"<b>Risk: {risk}</b> | Model Acc: {acc:.1%}"},
#         gauge={'bar': {'color': color}}
#     ))
#     st.plotly_chart(fig, use_container_width=True)

#     # MÉTRICAS
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Income", f"${data['income']:,}")
#     c2.metric("Deductions", f"${data['deductions']:,.0f}")
#     c3.metric("Tax", f"${data['tax']:,}")
#     c4.metric("Fraud Risk", f"{prob*100:.1f}%")

#     # ALERTA
#     if risk == "HIGH":
#         st.error("**HIGH FRAUD RISK** – Recommend full audit.")
#     elif risk == "MEDIUM":
#         st.warning("**MODERATE RISK** – Review deduction sources.")
#     else:
#         st.success("**LOW RISK** – Return appears clean.")

#     # REPORTE PDF
#     class PDF(FPDF):
#         def header(self):
#             self.set_font('Arial', 'B', 16)
#             self.cell(0, 10, 'IRS Fraud Report', ln=1, align='C')
#         def footer(self):
#             self.set_y(-15)
#             self.set_font('Arial', 'I', 8)
#             self.cell(0, 10, f'Page {self.page_no()}', align='C')

#     pdf = PDF()
#     pdf.add_page()
#     pdf.set_font('Arial', '', 12)
#     pdf.cell(0, 10, f"ZIP: {data['zipcode']} | Households: {data['num_returns']}", ln=1)
#     pdf.cell(0, 10, f"Income: ${data['income']:,} | Deductions: ${data['deductions']:,.0f}", ln=1)
#     pdf.cell(0, 10, f"Fraud Risk: {prob*100:.1f}% | Risk: {risk}", ln=1)
#     pdf_output = pdf.output(dest='S').encode('latin1')
#     b64 = base64.b64encode(pdf_output).decode()
#     href = f'<a href="data:application/pdf;base64,{b64}" download="fraud_report.pdf">Download PDF Report</a>'
#     st.markdown(href, unsafe_allow_html=True)

# # ------------------ INSIGHTS ------------------
# with st.expander("Fraud Intelligence Dashboard", expanded=False):
#     col1, col2 = st.columns(2)
#     with col1:
#         fig = px.histogram(df, x="income", nbins=50, title="Income Distribution")
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         sample = df.sample(1000)
#         fig = px.scatter(sample, x="income", y="deductions", color="is_fraud",
#                          color_discrete_map={0: "#22c55e", 1: "#ef4444"},
#                          hover_data=["zipcode"], title="Fraud Pattern")
#         st.plotly_chart(fig, use_container_width=True)

#     st.markdown("### Top 10 High-Risk ZIP Codes")
#     top_zips = df[df["is_fraud"] == 1]["zipcode"].value_counts().head(10)
#     st.dataframe(pd.DataFrame({"ZIP": top_zips.index.astype(str), "Fraud Cases": top_zips.values}))

# # ------------------ FOOTER ------------------
# st.markdown("""
# <div class="footer">
#     <strong>Built by Liliana Bustamante</strong> | CPA Candidate | J.D. Law | 28 AI Certifications<br>
#     <a href="https://irs.gov" target="_blank">IRS SOI 2021</a> • 
#     <a href="https://github.com/Moly-malibu" target="_blank">GitHub</a>
# </div>
# """, unsafe_allow_html=True)