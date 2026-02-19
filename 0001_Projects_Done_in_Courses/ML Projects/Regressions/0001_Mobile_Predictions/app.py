import streamlit as st
import joblib
import numpy as np

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('linear_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

mobile_model, mobile_scaler = load_assets()

st.title("📱 Mobile Price Prediction")
st.markdown("Enter specifications based on the model's training range:")

col1, col2 = st.columns(2)

with col1:
    
    weight = st.slider("Weight (grams)", 60.0, 750.0, 170.0)
    
    screen_size = st.slider("Screen Size (Inches)", 1.0, 13.0, 5.2)
    ppi = st.slider("PPI (Pixels Per Inch)", 120, 800, 335)
    cpu_core = st.selectbox("CPU Cores", [0, 1, 2, 4, 6, 8], index=3)
    cpu_freq = st.slider("CPU Frequency (GHz)", 0.0, 3.0, 1.5)

with col2:
    internal_mem = st.selectbox("Internal Memory (GB)", [4, 8, 16, 32, 64, 128], index=3)
    ram = st.slider("RAM (GB)", 0.5, 6.0, 2.0)
    rear_cam = st.slider("Rear Camera (MP)", 0, 25, 10)
    front_cam = st.slider("Front Camera (MP)", 0, 20, 5)
    battery = st.slider("Battery (mAh)", 800, 10000, 3000)
    thickness = st.slider("Thickness (mm)", 5.0, 19.0, 9.0)

st.divider()

if st.button("Predict Price", type="primary"):
    # The order must be exactly: 
    # weight, resoloution, ppi, cpu core, cpu freq, internal mem, ram, RearCam, Front_Cam, battery, thickness
    mobile_input = np.array([[
        weight, screen_size, ppi, cpu_core, cpu_freq, 
        internal_mem, ram, rear_cam, front_cam, battery, thickness
    ]])
    
    try:
        # 1. Scale
        scaled_input = mobile_scaler.transform(mobile_input)
        
        # 2. Predict
        prediction = mobile_model.predict(scaled_input)
        
        # 3. Handle result (linear regression can still be slightly negative for very low specs)
        price = max(0, prediction[0]) # UI safety clip
        
        st.success(f"### Estimated Price: ${price:.2f}")
        
    except Exception as e:
        st.error(f"Error: {e}")