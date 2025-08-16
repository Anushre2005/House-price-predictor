import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib, pandas as pd, json, os
from  src.utils import friendly_to_model_inputs, predict_one

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load model + metadata
pipe = joblib.load(os.path.join("artifacts", "model.joblib"))
meta = json.load(open(os.path.join("artifacts", "metrics.json")))

st.title("🏠 House Price Predictor")
st.caption("Task: Regression • Model: XGBoost • Target: House Price (USD)")

st.markdown("Enter the details below to estimate the house price:")

# --- USER-FRIENDLY INPUTS ---
income = st.slider("💰 Household Income (in $)", 10000, 200000, 65000, step=5000)

house_age = st.slider("🏚️ Age of House (years)", 1, 100, 20)

rooms = st.slider("🛋️ Average Rooms per Household", 1, 10, 5)

bedrooms = st.slider("🛏️ Average Bedrooms per Household", 1, 5, 2)

population = st.number_input("👨‍👩‍👧 Population in District", min_value=100, max_value=10000, value=1500, step=100)

occupants = st.slider("👥 Average Occupants per Household", 1, 10, 3)

latitude = st.selectbox("🌍 Latitude Region", options=[34.0, 35.0, 36.0, 37.0, 38.0], index=0)

longitude = st.selectbox("📍 Longitude Region", options=[-120.0, -119.0, -118.0, -117.0], index=2)

# Pack user inputs
user_input = {
    "income": income,
    "house_age": house_age,
    "rooms": rooms,
    "bedrooms": bedrooms,
    "population": population,
    "occupants": occupants,
    "latitude": latitude,
    "longitude": longitude,
}

# Predict button
if st.button("🔮 Predict House Price"):
    features = friendly_to_model_inputs(user_input)
    prediction = predict_one(pipe, features, task="regression", target="MedHouseVal")
    st.success(f"🏠 Estimated Price: ${prediction:,.0f}")
