import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the Model and Scaler
@st.cache_resource
def load_assets():
    # Streamlit looks for these in your root GitHub folder
    model = joblib.load('linear_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📱 Mobile Price Prediction")
st.write("Estimate the price based on hardware specifications.")

# 2. Input Fields (Based on your 'Cellphone.csv' ranges)
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input("Weight (grams)", 60.0, 750.0, 170.0)
    # Note: 'resoloution' in your dataset is screen size in inches
    resoloution = st.number_input("Screen Size (Inches)", 1.0, 13.0, 5.2)
    ppi = st.number_input("PPI", 120, 800, 335)
    cpu_core = st.selectbox("CPU Cores", [0, 1, 2, 4, 6, 8], index=5)
    cpu_freq = st.number_input("CPU Frequency (GHz)", 0.0, 3.0, 1.5)

with col2:
    internal_mem = st.number_input("Internal Memory (GB)", 2, 512, 32)
    ram = st.number_input("RAM (GB)", 0.0, 8.0, 2.0)
    RearCam = st.number_input("Rear Camera (MP)", 0.0, 30.0, 10.0)
    Front_Cam = st.number_input("Front Camera (MP)", 0.0, 20.0, 5.0)
    battery = st.number_input("Battery (mAh)", 500, 10000, 3000)
    thickness = st.number_input("Thickness (mm)", 5.0, 20.0, 9.0)

# 3. Prediction Logic
if st.button("Predict Price", type="primary"):
    # Feature order MUST match your training data columns exactly
    features = np.array([[
        weight, resoloution, ppi, cpu_core, cpu_freq, 
        internal_mem, ram, RearCam, Front_Cam, battery, thickness
    ]])
    
    try:
        # Apply the scaler first
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        # Ensure price isn't negative due to linear regression logic
        final_price = max(0, prediction[0])
        st.success(f"### Predicted Mobile Price: ${final_price:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")