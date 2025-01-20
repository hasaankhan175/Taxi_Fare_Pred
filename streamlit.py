import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Total Fare Predictor", page_icon="🚕", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚕 Total Fare Prediction App")
st.markdown("Predict the total fare of your ride with advanced analytics and an engaging interface!")

# Input Section
st.header("Input Trip Details")
trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, max_value=300, step=1)
distance_traveled = st.number_input("Distance Traveled (km)", min_value=0.1, max_value=500.0, step=0.1)
num_of_passengers = st.slider("Number of Passengers", 1, 6, 1)
tip = st.slider("Tip (INR)", 0, 200, 0)
miscellaneous_fees = st.number_input("Miscellaneous Fees (INR)", min_value=0.0, max_value=50.0, step=0.1)
surge_applied = st.selectbox("Surge Applied", ["Yes", "No"])

# Load Trained Model
@st.cache_resource
def load_model():
    with open(r"Random Forest Regressor___fare_pred.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Feature Preparation
surge_multiplier = 1.5 if surge_applied == "Yes" else 1.0
input_data = pd.DataFrame([{
    "trip_duration": trip_duration,
    "distance_traveled": distance_traveled,
    "num_of_passengers": num_of_passengers,
    "tip": tip,  # Use 'predicted_tip' instead of 'tip'
    "miscellaneous_fees": miscellaneous_fees,
    "surge_applied": surge_multiplier
}])

# Make Prediction
# Make Prediction
predicted_total_fare = model.predict(input_data)[0]  # Extract the first element

# Display Output
st.subheader("Predicted Total Fare")
st.metric("Total Fare", f"INR{predicted_total_fare:.2f}")

# Footer
st.markdown("""
    ---
    **Created with ❤️ by [Hasaan Khan]**  
    Predict total fares for your trips with ease and accuracy!
""")
