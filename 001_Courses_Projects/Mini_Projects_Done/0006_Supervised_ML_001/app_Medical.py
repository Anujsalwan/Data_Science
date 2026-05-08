
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Medical Insurance Cost Predictor")
st.caption(
    "Estimate annual medical insurance charges from a few personal details. "
    "The model is a tuned Gradient Boosting Regressor trained on the "
    "Health Insurance dataset."
)

# ------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------
@st.cache_resource
def load_model(path="best_model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_model_name(path="best_model_name.txt"):
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "Gradient Boosting"


model = load_model()
model_name = load_model_name()

if model is None:
    st.error(
        "⚠️ `best_model.pkl` not found. "
        "Run `python medical_insurance_prediction.py` first to train and save the model."
    )
    st.stop()

st.success(f"Loaded model: **Tuned {model_name}**")

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Enter patient details")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0, step=0.1,
            help="Body Mass Index = weight(kg) / height(m)²",
        )
        children = st.slider(
            "Number of children/dependents", min_value=0, max_value=10, value=0, step=1
        )

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox(
            "Region", ["northeast", "northwest", "southeast", "southwest"]
        )

    submitted = st.form_submit_button("Predict charge", use_container_width=True)

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "age":      age,
        "sex":      sex,
        "bmi":      bmi,
        "children": children,
        "smoker":   smoker,
        "region":   region,
    }])

    # Pipeline outputs log1p(charges); invert with expm1
    pred_log = model.predict(input_df)[0]
    pred = float(np.expm1(pred_log))

    st.markdown("---")
    st.subheader("Predicted annual insurance charge")
    st.metric(label="Estimated charge (USD)", value=f"${pred:,.2f}")

    # Helpful context
    if smoker == "yes":
        st.info(
            "Smoking is the largest single driver of charges in this dataset — "
            "smokers typically pay ~3–4× more than non-smokers, all else equal."
        )
    if bmi >= 30:
        st.info(
            "BMI ≥ 30 (obese range) tends to push charges up further, "
            "especially when combined with smoking."
        )

    with st.expander("Show input as sent to model"):
        st.dataframe(input_df, use_container_width=True)

# ------------------------------------------------------------
# Sidebar — model card
# ------------------------------------------------------------
with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"""
- **Algorithm:** Tuned **{model_name}** Regressor
- **Target:** `charges` (log-transformed during training, reverted for display)
- **Features:** age, sex, BMI, children, smoker, region
- **Preprocessing:** `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- **Tuning:** `GridSearchCV` with 5-fold CV
"""
    )
    if os.path.exists("outputs/model_comparison.csv"):
        st.subheader("Model comparison")
        cmp = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(cmp, use_container_width=True, hide_index=True)
