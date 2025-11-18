# utils/components.py
import streamlit as st
import plotly.graph_objects as go

def show_risk_gauge(prob, risk):
    color = "#dc2626" if risk == "HIGH" else "#f59e0b" if risk == "MEDIUM" else "#10b981"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': f"Risk: {risk}"}, gauge={'bar': {'color': color}}))
    st.plotly_chart(fig, use_container_width=True)

def show_metrics(income, ded, tax, prob):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"${income:,}")
    c2.metric("Deductions", f"${ded:,.0f}")
    c3.metric("Tax", f"${tax:,}")
    c4.metric("Risk", f"{prob*100:.1f}%")

def show_alert(risk):
    if risk == "HIGH": st.error("HIGH RISK")
    elif risk == "MEDIUM": st.warning("MEDIUM RISK")
    else: st.success("LOW RISK")

def download_report(data, prob, risk):
    report = f"Tax Report\nRisk: {risk}\nProbability: {prob*100:.1f}%"
    st.download_button("Download", report, "report.txt")
