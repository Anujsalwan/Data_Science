import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. Helper Function to generate/load the Scaler & Encoders
@st.cache_resource
def prepare_assets():
    # Load the trained classification model
    model = joblib.load('001_Courses_Projects/ML_Projects/Supervised_ML/Income_Classification/best_gradient_boosting_model.pkl')

    # Load CSV to recreate the scaler & label encoders
    # (model was trained on encoded + scaled data)
    df = pd.read_csv('001_Courses_Projects/ML_Projects/Supervised_ML/Income_Classification/income_evaluation.csv')

    # Strip whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Replace '?' (the dataset's missing-value marker) with NaN, then impute
    df = df.replace('?', np.nan)

    # Identify column groups
    target_col = 'income'
    categorical_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    numeric_cols = [
        'age', 'fnlwgt', 'education-num',
        'capital-gain', 'capital-loss', 'hours-per-week'
    ]

    # Imputation: mode for categoricals, median for numerics
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fit a LabelEncoder per categorical column (matches the typical notebook pipeline)
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target separately (so we can map prediction back to label)
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

    # Prepare features in the exact order used at training time
    feature_columns = numeric_cols + categorical_cols
    X = df[feature_columns]

    # Fit the scaler on all features
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, encoders, target_encoder, feature_columns

# Initialize Model, Scaler, and Encoders
try:
    model, scaler, encoders, target_encoder, feature_columns = prepare_assets()
except Exception as e:
    st.error(
        f"Error loading assets: {e}. Ensure 'best_gradient_boosting_model.pkl' "
        f"and 'income_evaluation.csv' are in the folder."
    )
    st.stop()

st.set_page_config(page_title="Income Classification Predictor", layout="wide")
st.title("💰 Income Classification App")
st.write(
    "Enter demographic and employment indicators to predict whether annual income "
    "is **>50K** or **≤50K** (UCI Adult / Kaggle Income Classification dataset)."
)

# 2. Input Fields Layout
st.subheader("Input Parameters")
col1, col2, col3 = st.columns(3)

# Helper: pull the original class labels from each fitted encoder so the
# selectboxes show human-readable options (not encoded integers).
def options(col):
    return list(encoders[col].classes_)

with col1:
    age = st.number_input("Age", 17, 100, 39)
    workclass = st.selectbox("Workclass", options('workclass'),
                             index=options('workclass').index('Private')
                             if 'Private' in options('workclass') else 0)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 0, 2000000, 77516)
    education = st.selectbox("Education", options('education'),
                             index=options('education').index('Bachelors')
                             if 'Bachelors' in options('education') else 0)
    education_num = st.number_input("Education-Num (years of edu)", 1, 20, 13)

with col2:
    marital_status = st.selectbox("Marital Status", options('marital-status'))
    occupation = st.selectbox("Occupation", options('occupation'))
    relationship = st.selectbox("Relationship", options('relationship'))
    race = st.selectbox("Race", options('race'))
    sex = st.selectbox("Sex", options('sex'))

with col3:
    capital_gain = st.number_input("Capital Gain (USD)", 0, 100000, 2174)
    capital_loss = st.number_input("Capital Loss (USD)", 0, 100000, 0)
    hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", options('native-country'),
                                  index=options('native-country').index('United-States')
                                  if 'United-States' in options('native-country') else 0)

# 3. Prediction Logic
if st.button("Predict Income Class", type="primary"):
    # 1. Encode categorical inputs using the fitted encoders
    try:
        encoded_categoricals = {
            'workclass': encoders['workclass'].transform([workclass])[0],
            'education': encoders['education'].transform([education])[0],
            'marital-status': encoders['marital-status'].transform([marital_status])[0],
            'occupation': encoders['occupation'].transform([occupation])[0],
            'relationship': encoders['relationship'].transform([relationship])[0],
            'race': encoders['race'].transform([race])[0],
            'sex': encoders['sex'].transform([sex])[0],
            'native-country': encoders['native-country'].transform([native_country])[0],
        }
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        st.stop()

    # 2. Assemble the row in the SAME column order used during training
    input_row = {
        'age': age,
        'fnlwgt': fnlwgt,
        'education-num': education_num,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        **encoded_categoricals,
    }
    features_df = pd.DataFrame([[input_row[c] for c in feature_columns]],
                               columns=feature_columns)

    try:
        # 3. Apply scaling
        scaled_features = scaler.transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns)

        # 4. Predict
        prediction = model.predict(scaled_features_df)
        pred_label = target_encoder.inverse_transform(prediction)[0]

        # 5. Probability (if the model supports it)
        prob_text = ""
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_features_df)[0]
            classes = target_encoder.inverse_transform(model.classes_)
            prob_map = dict(zip(classes, proba))
            prob_text = " | ".join([f"{cls}: {p*100:.1f}%" for cls, p in prob_map.items()])

        # Display Result
        if str(pred_label).strip() == '>50K':
            st.success(f"### Predicted Income: **{pred_label}** 💵")
        else:
            st.info(f"### Predicted Income: **{pred_label}**")

        if prob_text:
            st.caption(f"Class probabilities — {prob_text}")

            # Progress bar showing probability of >50K
            high_income_prob = next((p for cls, p in prob_map.items()
                                     if str(cls).strip() == '>50K'), 0.0)
            st.progress(min(max(float(high_income_prob), 0.0), 1.0))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
