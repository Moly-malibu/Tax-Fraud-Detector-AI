# utils/data_loader.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_tax_data():
    np.random.seed(42)
    n = 5000
    zipcode = np.random.randint(10000, 99999, n)
    income = np.random.lognormal(11, 0.8, n) * np.random.poisson(80, n)
    income = np.clip(income, 10000, 5_000_000)
    ded_ratio = np.random.uniform(0.05, 0.6, n)
    deductions = np.clip(income * ded_ratio, 0, income * 0.8)
    tax = np.clip((income - deductions) * np.random.uniform(0.15, 0.28, n), 0, None)
    df = pd.DataFrame({"zipcode": zipcode, "income": income, "deductions": deductions, "tax": tax})
    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / (df["income"] + 1)
    df["is_fraud"] = (((df["ded_ratio"] > 0.5) | (df["tax_ratio"] < 0.05)) & (np.random.rand(n) < 0.35)).astype(int)
    return df

@st.cache_data
def generate_cc_data():
    np.random.seed(123)
    n = 8000
    amount = np.clip(np.random.lognormal(4.2, 1.0, n), 1, 10000)
    time_since_last = np.clip(np.random.exponential(300, n), 0, 86400)
    merchant_cat = np.random.choice(["grocery", "online", "travel", "fuel", "restaurant", "other"], n)
    distance_km = np.clip(np.random.gamma(2, 15, n), 0, 500)
    df = pd.DataFrame({"amount": amount, "time_since_last": time_since_last, "merchant_cat": merchant_cat, "distance_km": distance_km})
    df = pd.get_dummies(df, columns=["merchant_cat"], prefix="cat")
    df["amt_per_sec"] = df["amount"] / (df["time_since_last"] + 1)
    df["amt_per_km"] = df["amount"] / (df["distance_km"] + 1)
    df["is_fraud"] = ((df["amount"] > 3000) | (df["time_since_last"] < 60) | (df["amt_per_sec"] > 5) & (np.random.rand(n) < 0.28)).astype(int)
    return df

@st.cache_data
def generate_insurance_data():
    np.random.seed(999)
    n = 6000
    age = np.random.randint(18, 85, n)
    vehicle_value = np.clip(np.random.lognormal(9.5, 0.7, n), 5000, 150000)
    claim_amount = np.clip(np.random.lognormal(7.8, 1.1, n), 100, 100000)
    df = pd.DataFrame({"age": age, "vehicle_value": vehicle_value, "claim_amount": claim_amount})
    df["claim_to_value"] = df["claim_amount"] / (df["vehicle_value"] + 1)
    df["is_fraud"] = ((df["claim_to_value"] > 0.9) | (df["claim_amount"] > 50000) & (np.random.rand(n) < 0.32)).astype(int)
    return df