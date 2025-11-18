# pages/_3_Insurance.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Mock data generator (replace with your own function if you have it)
def generate_insurance_data():
    np.random.seed(42)
    n = 500
    data = {
        "age": np.random.randint(18, 85, n),
        "vehicle_value": np.random.randint(5000, 150000, n),
        "claim_amount": np.random.randint(100, 100000, n),
    }
    df = pd.DataFrame(data)
    df["claim_to_value"] = df["claim_amount"] / (df["vehicle_value"] + 1)
    df["is_fraud"] = np.random.binomial(1, 0.10, n)
    return df

from sklearn.ensemble import RandomForestClassifier

def train_models(train_df, val_df, test_df):
    # For insurance: features are below
    features = ["age", "vehicle_value", "claim_amount", "claim_to_value"]
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(train_df[features], train_df["is_fraud"])
    # Replicate your tuple unpacking order: (None, None, model, None, features)
    return None, None, rf, None, features

def show_risk_gauge_plotly(prob, risk):
    color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "pink"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_alert(level):
    if level == "HIGH":
        st.error("⚠️ Fraud likely!")
    elif level == "MEDIUM":
        st.warning("Review recommended. Possible fraud.")
    else:
        st.success("Claim appears legit.")

def main():
    st.header("Insurance Claim Fraud Detection")
    df = generate_insurance_data()
    _, _, ins_model, _, ins_features = train_models(df, df, df)

    with st.sidebar:
        st.subheader("Claim")
        age = st.slider("Age", 18, 85, 38)
        vehicle_value = st.slider("Vehicle ($)", 5000, 150000, 35000)
        claim_amount = st.slider("Claim ($)", 100, 100000, 8500)
        if st.button("Check Fraud"):
            st.session_state.ins_input = {
                "age": age,
                "vehicle_value": vehicle_value,
                "claim_amount": claim_amount
            }

    if "ins_input" in st.session_state:
        data = st.session_state.ins_input
        X = pd.DataFrame([{
            "age": data["age"],
            "vehicle_value": data["vehicle_value"],
            "claim_amount": data["claim_amount"],
            "claim_to_value": data["claim_amount"] / (data["vehicle_value"] + 1)
        }])
        X = X.reindex(columns=ins_features, fill_value=0)
        st.write("Model input:", X)

        try:
            prob = ins_model.predict_proba(X)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prob = 0.0
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        show_risk_gauge_plotly(prob, risk)
        c1, c2, c3 = st.columns(3)
        c1.metric("Claim", f"${data['claim_amount']}")
        c2.metric("Vehicle", f"${data['vehicle_value']}")
        c3.metric("Risk", f"{prob*100:.1f}%")
        show_alert(risk)
    else:
        st.info("Enter claim details on the left and press 'Check Fraud'.")

if __name__ == "__main__":
    main()




# # pages/_3_Insurance.py
# import streamlit as st
# import pandas as pd
# from utils.data_loader import generate_insurance_data
# from utils.models import train_models
# from utils.components import show_risk_gauge, show_metrics, show_alert

# def main():
#     st.header("Insurance Claim Fraud Detection")
#     df = generate_insurance_data()
#     _, _, ins_model, _, ins_features = train_models(df, df, df)

#     with st.sidebar:
#         st.subheader("Claim")
#         age = st.slider("Age", 18, 85, 38)
#         vehicle_value = st.slider("Vehicle ($)", 5000, 150000, 35000)
#         claim_amount = st.slider("Claim ($)", 100, 100000, 8500)
#         if st.button("Check Fraud"):
#             st.session_state.ins_input = {
#                 "age": age, "vehicle_value": vehicle_value, "claim_amount": claim_amount
#             }

#     if "ins_input" in st.session_state:
#         data = st.session_state.ins_input
#         X = pd.DataFrame([{
#             "age": data["age"], "vehicle_value": data["vehicle_value"],
#             "claim_amount": data["claim_amount"],
#             "claim_to_value": data["claim_amount"] / (data["vehicle_value"] + 1)
#         }])
#         X = X.reindex(columns=ins_features, fill_value=0)
#         prob = ins_model.predict_proba(X)[0][1]
#         risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
#         show_risk_gauge(prob, risk)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Claim", f"${claim_amount}")
#         c2.metric("Vehicle", f"${vehicle_value}")
#         c3.metric("Risk", f"{prob*100:.1f}%")
#         show_alert(risk)