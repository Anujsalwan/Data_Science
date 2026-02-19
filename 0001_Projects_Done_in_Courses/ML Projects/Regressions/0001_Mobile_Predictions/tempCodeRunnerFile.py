import streamlit as st
import joblib
import numpy as np
## Load the trained model
model = joblib.load('linear_model.pkl')

# Input fields for user to enter mobile specifications
st.title("Mobile Price Prediction")
ram = st.number_input("RAM (in GB)", min_value=1, max_value=16, step=1)
storage = st.number_input("Storage (in GB)", min_value=8, max_value=512, step=8)
battery = st.number_input("Battery Capacity (in mAh)", min_value=1000, max_value=10000, step=100)
camera = st.number_input("Camera Resolution (in MP)", min_value=1, max_value=108, step=1)
processor_speed = st.number_input("Processor Speed (in GHz)", min_value=0.5, max_value=5.0, step=0.1)
# Predict button
if st.button("Predict Price"):
    # Create a feature array from user inputs
    features = np.array([[ram, storage, battery, camera, processor_speed]])
    # Predict the price using the loaded model
    predicted_price = model.predict(features)
    st.success(f"The predicted price of the mobile is: ${predicted_price[0]:.2f}")        
    


