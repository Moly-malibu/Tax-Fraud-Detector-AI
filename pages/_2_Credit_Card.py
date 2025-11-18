# pages/_2_Credit_Card.py
# # pages/_2_Credit_Card.py

# pages/_2_Credit_Card.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def generate_cc_data():
    np.random.seed(42)
    data = {
        "amount": np.random.randint(1, 4000, 500),
        "time_since_last": np.random.randint(0, 86400, 500),
        "distance_km": np.random.uniform(0, 150, 500),
        "cat_grocery": np.random.binomial(1, 0.2, 500),
        "cat_online": np.random.binomial(1, 0.2, 500),
        "cat_travel": np.random.binomial(1, 0.1, 500),
        "cat_fuel": np.random.binomial(1, 0.15, 500),
        "cat_restaurant": np.random.binomial(1, 0.15, 500),
        "cat_other": np.random.binomial(1, 0.2, 500),
    }
    df = pd.DataFrame(data)
    df["amt_per_sec"] = df["amount"] / (df["time_since_last"] + 1)
    df["amt_per_km"] = df["amount"] / (df["distance_km"] + 1)
    df["is_fraud"] = np.random.binomial(1, 0.08, 500)
    return df

from sklearn.ensemble import RandomForestClassifier

def train_models(train_df, val_df, test_df):
    features = [
        "amount", "time_since_last", "distance_km",
        "amt_per_sec", "amt_per_km",
        "cat_grocery", "cat_online", "cat_travel",
        "cat_fuel", "cat_restaurant", "cat_other"
    ]
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(train_df[features], train_df["is_fraud"])
    return None, rf, None, features, None

def show_risk_gauge_plotly(prob):
    color = "red" if prob > 0.5 else "green"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "pink"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_alert(level):
    if level == "HIGH":
        st.error("⚠️ Fraud likely!")
    else:
        st.success("Transaction appears legit.")

def main():
    st.header("Credit Card Fraud Detection")

    df = generate_cc_data()
    _, cc_model, _, cc_features, _ = train_models(df, df, df)

    with st.sidebar:
        st.subheader("Transaction")
        amount = st.slider("Amount ($)", 1, 10000, 89)
        time_since_last = st.slider("Sec since last", 0, 86400, 300)
        merchant_cat = st.selectbox("Merchant", ["grocery", "online", "travel", "fuel", "restaurant", "other"])
        distance_km = st.slider("Distance (km)", 0, 500, 12)
        check = st.button("Check Fraud")
        if check:
            st.session_state.cc_input = {
                "amount": amount,
                "time_since_last": time_since_last,
                "merchant_cat": merchant_cat,
                "distance_km": distance_km
            }

    if "cc_input" in st.session_state:
        data = st.session_state.cc_input
        input_dict = {
            "amount": data["amount"],
            "time_since_last": data["time_since_last"],
            "distance_km": data["distance_km"],
            "amt_per_sec": data["amount"] / (data["time_since_last"] + 1),
            "amt_per_km": data["amount"] / (data["distance_km"] + 1),
            "cat_grocery": 1.0 if data["merchant_cat"] == "grocery" else 0.0,
            "cat_online": 1.0 if data["merchant_cat"] == "online" else 0.0,
            "cat_travel": 1.0 if data["merchant_cat"] == "travel" else 0.0,
            "cat_fuel": 1.0 if data["merchant_cat"] == "fuel" else 0.0,
            "cat_restaurant": 1.0 if data["merchant_cat"] == "restaurant" else 0.0,
            "cat_other": 1.0 if data["merchant_cat"] == "other" else 0.0,
        }
        df_input = pd.DataFrame([input_dict])
        df_input = df_input.reindex(columns=cc_features, fill_value=0)

        # Show input for debug
        st.write("Model input:", df_input)

        try:
            prob = cc_model.predict_proba(df_input)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prob = 0.0

        risk = "FRAUD" if prob > 0.5 else "LEGIT"
        show_risk_gauge_plotly(prob)
        c1, c2, c3 = st.columns(3)
        c1.metric("Amount", f"${data['amount']}")
        c2.metric("Time", f"{data['time_since_last']}s")
        c3.metric("Risk", f"{prob*100:.1f}%")
        show_alert("HIGH" if risk == "FRAUD" else "LOW")
    else:
        st.info("Please enter transaction details and press 'Check Fraud'.")

if __name__ == "__main__":
    main()





# import streamlit as st
# import pandas as pd
# import numpy as np

# # --------------------------
# # Simulated data and model - REPLACE with your utils functions if available!
# def generate_cc_data():
#     np.random.seed(42)
#     data = {
#         "amount": np.random.randint(1, 4000, 500),
#         "time_since_last": np.random.randint(0, 86400, 500),
#         "distance_km": np.random.uniform(0, 150, 500),
#         "cat_grocery": np.random.binomial(1, 0.2, 500),
#         "cat_online": np.random.binomial(1, 0.2, 500),
#         "cat_travel": np.random.binomial(1, 0.1, 500),
#         "cat_fuel": np.random.binomial(1, 0.15, 500),
#         "cat_restaurant": np.random.binomial(1, 0.15, 500),
#         "cat_other": np.random.binomial(1, 0.2, 500),
#     }
#     df = pd.DataFrame(data)
#     df["amt_per_sec"] = df["amount"] / (df["time_since_last"] + 1)
#     df["amt_per_km"] = df["amount"] / (df["distance_km"] + 1)
#     df["is_fraud"] = np.random.binomial(1, 0.08, 500)
#     return df

# from sklearn.ensemble import RandomForestClassifier

# def train_models(train_df, val_df, test_df):
#     features = [
#         "amount", "time_since_last", "distance_km",
#         "amt_per_sec", "amt_per_km",
#         "cat_grocery", "cat_online", "cat_travel",
#         "cat_fuel", "cat_restaurant", "cat_other"
#     ]
#     rf = RandomForestClassifier(n_estimators=10, random_state=42)
#     rf.fit(train_df[features], train_df["is_fraud"])
#     return None, rf, None, features, None

# def show_risk_gauge(prob, risk):
#     st.write(f"### Risk Gauge: {risk} — {prob*100:.1f}%")

# def show_alert(level):
#     if level == "HIGH":
#         st.error("⚠️ Fraud likely!")
#     else:
#         st.success("Transaction appears legit.")

# # --------------------------

# def main():
#     st.header("Credit Card Fraud Detection")

#     df = generate_cc_data()
#     _, cc_model, _, cc_features, _ = train_models(df, df, df)

#     with st.sidebar:
#         st.subheader("Transaction")
#         amount = st.slider("Amount ($)", 1, 10000, 89)
#         time_since_last = st.slider("Sec since last", 0, 86400, 300)
#         merchant_cat = st.selectbox("Merchant", ["grocery", "online", "travel", "fuel", "restaurant", "other"])
#         distance_km = st.slider("Distance (km)", 0, 500, 12)
#         check = st.button("Check Fraud")
#         if check:
#             st.session_state.cc_input = {
#                 "amount": amount,
#                 "time_since_last": time_since_last,
#                 "merchant_cat": merchant_cat,
#                 "distance_km": distance_km
#             }

#     if "cc_input" in st.session_state:
#         data = st.session_state.cc_input
#         input_dict = {
#             "amount": data["amount"],
#             "time_since_last": data["time_since_last"],
#             "distance_km": data["distance_km"],
#             "amt_per_sec": data["amount"] / (data["time_since_last"] + 1),
#             "amt_per_km": data["amount"] / (data["distance_km"] + 1),
#             "cat_grocery": 1.0 if data["merchant_cat"] == "grocery" else 0.0,
#             "cat_online": 1.0 if data["merchant_cat"] == "online" else 0.0,
#             "cat_travel": 1.0 if data["merchant_cat"] == "travel" else 0.0,
#             "cat_fuel": 1.0 if data["merchant_cat"] == "fuel" else 0.0,
#             "cat_restaurant": 1.0 if data["merchant_cat"] == "restaurant" else 0.0,
#             "cat_other": 1.0 if data["merchant_cat"] == "other" else 0.0,
#         }
#         df_input = pd.DataFrame([input_dict])
#         df_input = df_input.reindex(columns=cc_features, fill_value=0)

#         # Show input for debug
#         st.write("Model input:", df_input)

#         try:
#             prob = cc_model.predict_proba(df_input)[0][1]
#         except Exception as e:
#             st.error(f"Prediction error: {e}")
#             prob = 0.0

#         risk = "FRAUD" if prob > 0.5 else "LEGIT"

#         show_risk_gauge(prob, risk)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Amount", f"${data['amount']}")
#         c2.metric("Time", f"{data['time_since_last']}s")
#         c3.metric("Risk", f"{prob*100:.1f}%")
#         show_alert("HIGH" if risk == "FRAUD" else "LOW")
#     else:
#         st.info("Please enter transaction details and press 'Check Fraud'.")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import pandas as pd
# import numpy as np

# # --------------------------
# # MOCK IMPLEMENTATIONS -- replace with your real ones!
# def generate_cc_data():
#     # Example fake dataset with 500 rows
#     np.random.seed(42)
#     data = {
#         "amount": np.random.randint(1, 4000, 500),
#         "time_since_last": np.random.randint(0, 86400, 500),
#         "distance_km": np.random.uniform(0, 150, 500),
#         "cat_grocery": np.random.binomial(1, 0.2, 500),
#         "cat_online": np.random.binomial(1, 0.2, 500),
#         "cat_travel": np.random.binomial(1, 0.1, 500),
#         "cat_fuel": np.random.binomial(1, 0.15, 500),
#         "cat_restaurant": np.random.binomial(1, 0.15, 500),
#         "cat_other": np.random.binomial(1, 0.2, 500),
#     }
#     df = pd.DataFrame(data)
#     df["amt_per_sec"] = df["amount"]/(df["time_since_last"]+1)
#     df["amt_per_km"] = df["amount"]/(df["distance_km"]+1)
#     df["is_fraud"] = np.random.binomial(1, 0.08, 500)
#     return df

# from sklearn.ensemble import RandomForestClassifier

# def train_models(train_df, val_df, test_df):
#     # Model expects: all features (including one-hot categories), output
#     features = [
#         "amount", "time_since_last", "distance_km", "amt_per_sec", "amt_per_km",
#         "cat_grocery", "cat_online", "cat_travel", "cat_fuel", "cat_restaurant", "cat_other"
#     ]
#     rf = RandomForestClassifier(n_estimators=10, random_state=42)
#     rf.fit(train_df[features], train_df["is_fraud"])
#     # Return (dummy values for other unused return slots)
#     return None, rf, None, features, None

# def show_risk_gauge(prob, risk):
#     st.write(f"### Risk Gauge: {risk} ({prob*100:.1f}%)")

# def show_alert(level):
#     if level == "HIGH":
#         st.error("⚠️ Fraud likely!")
#     else:
#         st.success("Transaction appears legit.")

# # --------------------------

# def main():
#     st.header("Credit Card Fraud Detection")

#     # Load or simulate data
#     df = generate_cc_data()

#     # Train or load model and features
#     _, cc_model, _, cc_features, _ = train_models(df, df, df)

#     # Sidebar Inputs
#     with st.sidebar:
#         st.subheader("Transaction")
#         amount = st.slider("Amount ($)", 1, 10000, 89)
#         time_since_last = st.slider("Sec since last", 0, 86400, 300)
#         merchant_cat = st.selectbox("Merchant", ["grocery", "online", "travel", "fuel", "restaurant", "other"])
#         distance_km = st.slider("Distance (km)", 0, 500, 12)
#         if st.button("Check Fraud"):
#             st.session_state.cc_input = {
#                 "amount": amount, 
#                 "time_since_last": time_since_last,
#                 "merchant_cat": merchant_cat, 
#                 "distance_km": distance_km
#             }

#     # Main logic
#     if "cc_input" in st.session_state:
#         st.write("Predicting based on:", st.session_state.cc_input)
#         data = st.session_state.cc_input
#         input_dict = {
#             "amount": data["amount"],
#             "time_since_last": data["time_since_last"],
#             "distance_km": data["distance_km"],
#             "amt_per_sec": data["amount"] / (data["time_since_last"] + 1),
#             "amt_per_km": data["amount"] / (data["distance_km"] + 1),
#             # Set all categories to 0, then 1 for selected
#             "cat_grocery": 1.0 if data["merchant_cat"] == "grocery" else 0.0,
#             "cat_online": 1.0 if data["merchant_cat"] == "online" else 0.0,
#             "cat_travel": 1.0 if data["merchant_cat"] == "travel" else 0.0,
#             "cat_fuel": 1.0 if data["merchant_cat"] == "fuel" else 0.0,
#             "cat_restaurant": 1.0 if data["merchant_cat"] == "restaurant" else 0.0,
#             "cat_other": 1.0 if data["merchant_cat"] == "other" else 0.0,
#         }
#         df_input = pd.DataFrame([input_dict])
#         df_input = df_input.reindex(columns=cc_features, fill_value=0)
#         st.write("Input to model:", df_input)

#         try:
#             prob = cc_model.predict_proba(df_input)[0][1]
#         except Exception as e:
#             st.error(f"Model prediction failed: {e}")
#             prob = 0.0
#         risk = "FRAUD" if prob > 0.5 else "LEGIT"

#         show_risk_gauge(prob, risk)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Amount", f"${amount}")
#         c2.metric("Time", f"{time_since_last}s")
#         c3.metric("Risk", f"{prob*100:.1f}%")
#         show_alert("HIGH" if risk == "FRAUD" else "LOW")
#     else:
#         st.info("Enter a transaction and click 'Check Fraud'.")

# # Required by Streamlit to set this as entrypoint for your page tab
# if __name__ == "__main__":
#     main()





# # pages/_2_Credit_Card.py
# import streamlit as st
# import pandas as pd
# from utils.data_loader import generate_cc_data
# from utils.models import train_models
# from utils.components import show_risk_gauge, show_metrics, show_alert

# def main():
#     st.header("Credit Card Fraud Detection")
#     df = generate_cc_data()
#     _, cc_model, _, cc_features, _ = train_models(df, df, df)

#     with st.sidebar:
#         st.subheader("Transaction")
#         amount = st.slider("Amount ($)", 1, 10000, 89)
#         time_since_last = st.slider("Sec since last", 0, 86400, 300)
#         merchant_cat = st.selectbox("Merchant", ["grocery", "online", "travel", "fuel", "restaurant", "other"])
#         distance_km = st.slider("Distance (km)", 0, 500, 12)
#         if st.button("Check Fraud"):
#             st.session_state.cc_input = {
#                 "amount": amount, "time_since_last": time_since_last,
#                 "merchant_cat": merchant_cat, "distance_km": distance_km
#             }

#     if "cc_input" in st.session_state:
#         data = st.session_state.cc_input
#         df_input = pd.DataFrame([{
#             "amount": data["amount"], "time_since_last": data["time_since_last"],
#             "distance_km": data["distance_km"], "amt_per_sec": data["amount"]/(data["time_since_last"]+1),
#             "amt_per_km": data["amount"]/(data["distance_km"]+1),
#             f"cat_{data['merchant_cat']}": 1.0
#         }])
#         df_input = df_input.reindex(columns=cc_features, fill_value=0)
#         prob = cc_model.predict_proba(df_input)[0][1]
#         risk = "FRAUD" if prob > 0.5 else "LEGIT"
#         show_risk_gauge(prob, risk)
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Amount", f"${amount}")
#         c2.metric("Time", f"{time_since_last}s")
#         c3.metric("Risk", f"{prob*100:.1f}%")
#         show_alert("HIGH" if risk == "FRAUD" else "LOW")