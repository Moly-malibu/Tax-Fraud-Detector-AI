# utils/styles.py
import streamlit as st

def apply_global_styles():
    st.markdown("""
    <style>
        .main {background:#f8fafc;}
        .hero {background:linear-gradient(90deg,#1e3a8a,#3b82f6); padding:2.5rem; border-radius:18px; color:white; text-align:center; margin-bottom:2rem; box-shadow:0 10px 25px rgba(0,0,0,0.15);}
        .hero h1 {margin:0; font-size:3rem; font-weight:800;}
        .hero .highlight {color:#fbbf24; font-weight:700;}
        .metric-box {background:#f1f5f9; padding:1rem; border-radius:12px; text-align:center; font-weight:600;}
        .footer {text-align:center; margin-top:3rem; color:#64748b; font-size:0.9rem;}
    </style>
    """, unsafe_allow_html=True)

def hero_banner():
    st.markdown("""
    <div class="hero">
        <h1>Fraud Detector AI</h1>
        <p style="font-size:1.3rem;"><strong>Tax • Credit Card • Insurance</strong> + API + Batch + SHAP</p>
        <p><span class="highlight">Real-time API</span> • <span class="highlight">CSV Batch</span> • <span class="highlight">Explainable AI</span></p>
        <p><strong>Liliana Bustamante</strong> | CPA Candidate | J.D. | 28 AI Certs</p>
    </div>
    """, unsafe_allow_html=True)