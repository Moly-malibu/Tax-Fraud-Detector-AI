# utils/models.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

@st.cache_resource
def train_models(tax_df, cc_df, ins_df):
    # Tax
    X_tax = tax_df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]
    y_tax = tax_df["is_fraud"]
    tax_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    tax_model.fit(X_tax, y_tax)

    # CC
    cc_features = [c for c in cc_df.columns if c != "is_fraud"]
    X_cc = cc_df[cc_features]
    y_cc = cc_df["is_fraud"]
    cc_model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    cc_model.fit(X_cc, y_cc)

    # Insurance
    ins_features = [c for c in ins_df.columns if c != "is_fraud"]
    X_ins = ins_df[ins_features]
    y_ins = ins_df["is_fraud"]
    ins_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    ins_model.fit(X_ins, y_ins)

    return tax_model, cc_model, ins_model, cc_features, ins_features

def save_models(tax_model, cc_model, ins_model):
    joblib.dump(tax_model, "models/tax_model.pkl")
    joblib.dump(cc_model, "models/cc_model.pkl")
    joblib.dump(ins_model, "models/ins_model.pkl")