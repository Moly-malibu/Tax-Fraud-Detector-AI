# pages/_6_Insights.py
import streamlit as st
import shap
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import generate_tax_data
from utils.models import train_models

def main():
    st.header("Model Insights & Explainability")
    df = generate_tax_data()
    model, _, _, _, _ = train_models(df, df, df)
    X = df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]].sample(50)

    # --- Feature Importance ---
    st.subheader("Feature Importance (Random Forest)")
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = go.Figure(go.Bar(
        x=imp["Importance"], y=imp["Feature"],
        orientation='h', marker_color='#3b82f6'
    ))
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- SHAP Values (SIN MATPLOTLIB) ---
    st.subheader("SHAP Impact (Interactive)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP promedio por feature
    shap_avg = pd.DataFrame({
        "Feature": X.columns,
        "SHAP": [abs(val).mean() for val in shap_values[1].T]
    }).sort_values("SHAP", ascending=False)

    fig = go.Figure(go.Bar(
        x=shap_avg["SHAP"], y=shap_avg["Feature"],
        orientation='h', marker_color='#ef4444'
    ))
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Force Plot INTERACTIVO (HTML) ---
    st.subheader("SHAP Force Plot (Click & Explore)")
    sample_idx = st.selectbox("Select a case", X.index)
    sample = X.loc[[sample_idx]]
    shap_val = explainer.shap_values(sample)[1]

    force_html = shap.force_plot(
        explainer.expected_value[1],
        shap_val[0],
        sample.iloc[0],
        show=False,
        matplotlib=False  # SIN MATPLOTLIB
    )
    st.components.v1.html(force_html.html(), height=220, scrolling=True)

    # --- Top ZIPs ---
    st.subheader("Top 10 High-Risk ZIP Codes")
    top = df[df["is_fraud"] == 1]["zipcode"].value_counts().head(10)
    if len(top) > 0:
        st.dataframe(pd.DataFrame({"ZIP": top.index.astype(str), "Fraud Cases": top.values}))
    else:
        st.info("No fraud detected in sample.")


# # pages/_6_Insights.py
# import streamlit as st
# import shap
# import matplotlib.pyplot as plt
# from utils.data_loader import generate_tax_data
# from utils.models import train_models

# def main():
#     st.header("Model Insights & Explainability")
#     df = generate_tax_data()
#     tax_model, _, _, _, _ = train_models(df, df, df)
#     X = df[["income", "deductions", "tax", "ded_ratio", "tax_ratio"]]

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Feature Importance")
#         imp = pd.DataFrame({
#             "Feature": X.columns,
#             "Importance": tax_model.feature_importances_
#         }).sort_values("Importance", ascending=False)
#         st.bar_chart(imp.set_index("Feature"))

#     with col2:
#         st.subheader("Top 10 Fraud ZIPs")
#         top = df[df["is_fraud"]==1]["zipcode"].value_counts().head(10)
#         st.dataframe(pd.DataFrame({"ZIP": top.index.astype(str), "Cases": top.values}))

#     if st.button("Generate SHAP Explanation (Sample)"):
#         explainer = shap.TreeExplainer(tax_model)
#         shap_vals = explainer.shap_values(X.sample(1))
#         fig, ax = plt.subplots()
#         shap.waterfall_plot(shap.Explanation(values=shap_vals[1][0], base_values=explainer.expected_value[1], data=X.sample(1).iloc[0]))
#         st.pyplot(fig)