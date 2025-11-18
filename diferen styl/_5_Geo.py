# pages/_5_Geo.py
import streamlit as st
import plotly.express as px
import pandas as pd
from utils.data_loader import generate_tax_data

def main():
    st.header("Geo-Fraud Heatmap")
    df = generate_tax_data()
    fraud_by_zip = df.groupby("zipcode")["is_fraud"].mean().reset_index()

    # Sample 50 ZIPs with lat/lon
    zip_data = pd.DataFrame({
        'zipcode': [90210,10001,60601,33101,77002,94102,30303,75201,98101,90001],
        'lat': [34.09,40.76,41.88,25.77,29.76,37.77,33.75,32.78,47.61,33.97],
        'lng': [-118.41,-73.99,-87.63,-80.19,-95.37,-122.42,-84.39,-96.80,-122.33,-118.24]
    })
    fraud_map = fraud_by_zip.merge(zip_data, on="zipcode", how="inner")

    fig = px.scatter_mapbox(
        fraud_map, lat="lat", lon="lng", size="is_fraud", color="is_fraud",
        hover_name="zipcode", color_continuous_scale="Reds", zoom=3,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig, use_container_width=True)