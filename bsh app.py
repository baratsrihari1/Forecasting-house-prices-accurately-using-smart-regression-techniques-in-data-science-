import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("xgboost_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="House Price Predictor", layout="centered")

# App Title
st.title("🏠 House Price Prediction App")
st.markdown("Use this app to predict house prices based on input features.")

# Sidebar for input
st.sidebar.header("Input Features")

def user_input_features():
    sqft = st.sidebar.slider("Living Area (sqft)", 500, 10000, 1500)
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 8, 2)
    floors = st.sidebar.slider("Number of Floors", 1, 3, 1)
    year_built = st.sidebar.slider("Year Built", 1900, 2023, 2000)
    location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])

    # Encode location manually (assuming Urban=2, Suburban=1, Rural=0)
    location_encoded = {"Urban": 2, "Suburban": 1, "Rural": 0}[location]
    house_age = 2025 - year_built
    total_rooms = bedrooms + bathrooms
    price_per_sqft = 0  # not known, usually calculated after price

    features = pd.DataFrame({
        "sqft_living": [sqft],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "floors": [floors],
        "location": [location_encoded],
        "house_age": [house_age],
        "total_rooms": [total_rooms]
    })
    return features

input_df = user_input_features()

# Show input
st.subheader("Input Data")
st.write(input_df)

# Preprocess input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict House Price"):
    prediction = model.predict(scaled_input)
    st.success(f"🏡 Predicted House Price: ${int(prediction[0]):,}")

# Footer
st.markdown("---")
st.caption("Built with 🧠 Machine Learning & Streamlit")
