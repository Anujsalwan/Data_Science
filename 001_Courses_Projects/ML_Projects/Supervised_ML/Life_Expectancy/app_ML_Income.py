import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. Helper Function to generate/load the Scaler and training columns from the CSV
@st.cache_resource
def prepare_assets():
    base_path = '001_Courses_Projects/ML_Projects/Supervised_ML/Life_Expectancy/'

    # Load the trained classifier
    model = joblib.load(os.path.join(base_path, 'best_xgboost_classifier.pkl'))

    # Load CSV to recreate the scaler and training columns
    df = pd.read_csv(os.path.join(base_path, 'adult.csv'))

    # Preprocessing to match the training pipeline in the notebook
    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop fnlwgt (dropped before encoding in the notebook)
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    # Separate target
    target_col = "income" if "income" in df.columns else df.columns[-1]
    df_clean = df.drop(columns=[target_col])

    # One-hot encode categorical columns (drop_first=True, same as training)
    cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    # Numeric columns the scaler was fit on
    numeric_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    # Fit the scaler on the numeric columns only
    scaler = StandardScaler()
    scaler.fit(X[numeric_cols])

    return model, scaler, X.columns.tolist(), numeric_cols

# Initialize Model, Scaler and feature columns
try:
    model, scaler, feature_columns, NUMERIC_COLS = prepare_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure 'best_xgboost_classifier.pkl' and 'adult.csv' are in the folder.")
    st.stop()

st.set_page_config(page_title="Income Prediction", layout="wide")
st.title("💰 Income Prediction App")
st.write("Enter demographic and employment details to predict whether annual income exceeds **$50K** (UCI Adult dataset).")

# 2. Dropdown options (taken directly from the training CSV)
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
    "State-gov", "Without-pay", "Never-worked", "?",
]
EDUCATION_OPTIONS = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
    "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
    "Masters", "Prof-school", "Doctorate",
]
EDUCATION_NUM_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}
MARITAL_OPTIONS = [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed",
    "Married-spouse-absent", "Married-AF-spouse",
]
OCCUPATION_OPTIONS = [
    "Prof-specialty", "Exec-managerial", "Adm-clerical", "Sales", "Craft-repair",
    "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners",
    "Tech-support", "Farming-fishing", "Protective-serv", "Priv-house-serv",
    "Armed-Forces", "?",
]
RELATIONSHIP_OPTIONS = [
    "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative",
]
RACE_OPTIONS = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX_OPTIONS = ["Male", "Female"]
COUNTRY_OPTIONS = [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico",
    "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China",
    "Italy", "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland",
    "Columbia", "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru",
    "Greece", "France", "Ecuador", "Ireland", "Hong", "Cambodia", "Trinadad&Tobago",
    "Laos", "Thailand", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Honduras",
    "Hungary", "Scotland", "Holand-Netherlands", "?",
]

# 3. Input Fields Layout
st.subheader("Input Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 17, 90, 37)
    workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS, index=0)
    education = st.selectbox("Education", EDUCATION_OPTIONS, index=EDUCATION_OPTIONS.index("HS-grad"))
    occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=0)
    hours_per_week = st.number_input("Hours per week", 1, 99, 40)

with col2:
    sex = st.selectbox("Sex", SEX_OPTIONS, index=0)
    race = st.selectbox("Race", RACE_OPTIONS, index=0)
    marital_status = st.selectbox("Marital status", MARITAL_OPTIONS, index=0)
    relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS, index=0)
    native_country = st.selectbox("Native country", COUNTRY_OPTIONS, index=0)

with col3:
    capital_gain = st.number_input("Capital gain", 0, 99999, 0, step=100)
    capital_loss = st.number_input("Capital loss", 0, 4356, 0, step=50)

# 4. Prediction Logic
if st.button("Predict Income", type="primary"):
    # Build raw input dict — keys MUST match the original CSV column names
    raw_input = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "education-num": EDUCATION_NUM_MAP[education],
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    try:
        # 1. Create DataFrame from raw input
        df = pd.DataFrame([raw_input])

        # 2. Strip whitespace from string columns (matches training)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # 3. One-hot encode (drop_first=True, same as training)
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # 4. Align columns to training feature space
        # (adds missing dummies as 0, drops extras, enforces column order)
        features_df = df.reindex(columns=feature_columns, fill_value=0)

        # 5. Scale numeric columns with the same scaler used in training
        cols_to_scale = [c for c in NUMERIC_COLS if c in features_df.columns]
        if cols_to_scale:
            features_df[cols_to_scale] = scaler.transform(features_df[cols_to_scale])

        # 6. Predict
        prediction = model.predict(features_df)[0]

        # Probability if available
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]

        # Display Result
        st.divider()
        if int(prediction) == 1:
            st.success("### 💸 Predicted Income: **> $50K**")
        else:
            st.info("### 💵 Predicted Income: **≤ $50K**")

        if proba is not None:
            pc1, pc2 = st.columns(2)
            pc1.metric("P(income ≤ $50K)", f"{proba[0]*100:.1f}%")
            pc2.metric("P(income > $50K)", f"{proba[1]*100:.1f}%")
            st.progress(min(max(float(proba[1]), 0.0), 1.0))

    except Exception as e:
        st.error(f"Prediction Error: {e}")