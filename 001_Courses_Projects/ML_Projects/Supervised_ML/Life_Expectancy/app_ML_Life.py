import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. Helper Function to recreate the EXACT preprocessing used during training
@st.cache_resource
def prepare_assets():
    # Load the trained classification model
    model = joblib.load('001_Courses_Projects/ML_Projects/Supervised_ML/Life_Expectancy/best_xgboost_classifier.pkl')

    # Load CSV to recreate the scaler & encoders
    df = pd.read_csv('001_Courses_Projects/ML_Projects/Supervised_ML/Life_Expectancy/income_evaluation.csv')

    # Strip whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Replace '?' with NaN, then impute
    df = df.replace('?', np.nan)

    target_col = 'income'
    categorical_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    # NOTE: model was trained WITHOUT 'fnlwgt' (per the error message)
    numeric_cols = [
        'age', 'education-num',
        'capital-gain', 'capital-loss', 'hours-per-week'
    ]

    # Imputation
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in numeric_cols + ['fnlwgt']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Save the unique categories for each categorical (for the dropdowns)
    category_options = {col: sorted(df[col].unique().tolist()) for col in categorical_cols}

    # Encode target
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

    # ---- ONE-HOT ENCODE features (this is what the model was trained on) ----
    # drop_first=True is what produces the column list seen in the error
    # (e.g. no 'workclass_?' since '?' was imputed, no 'sex_Female' since drop_first)
    X = df[numeric_cols + categorical_cols]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # The exact feature order the model expects
    feature_columns = list(X_encoded.columns)

    # Fit the scaler on the one-hot encoded data
    scaler = StandardScaler()
    scaler.fit(X_encoded)

    return model, scaler, target_encoder, feature_columns, category_options, numeric_cols, categorical_cols


# Initialize
try:
    (model, scaler, target_encoder, feature_columns,
     category_options, numeric_cols, categorical_cols) = prepare_assets()
except Exception as e:
    st.error(
        f"Error loading assets: {e}. Ensure 'best_xgboost_classifier.pkl' "
        f"and 'income_evaluation.csv' are in the folder."
    )
    st.stop()

st.set_page_config(page_title="Income Classification Predictor", layout="wide")
st.title("💰 Income Classification App")
st.write(
    "Enter demographic and employment indicators to predict whether annual income "
    "is **>50K** or **≤50K** (UCI Adult / Kaggle Income Classification dataset)."
)

# 2. Input Fields
st.subheader("Input Parameters")
col1, col2, col3 = st.columns(3)

def opts(col):
    return category_options[col]

with col1:
    age = st.number_input("Age", 17, 100, 39)
    workclass = st.selectbox("Workclass", opts('workclass'),
                             index=opts('workclass').index('Private')
                             if 'Private' in opts('workclass') else 0)
    education = st.selectbox("Education", opts('education'),
                             index=opts('education').index('Bachelors')
                             if 'Bachelors' in opts('education') else 0)
    education_num = st.number_input("Education-Num (years of edu)", 1, 20, 13)

with col2:
    marital_status = st.selectbox("Marital Status", opts('marital-status'))
    occupation = st.selectbox("Occupation", opts('occupation'))
    relationship = st.selectbox("Relationship", opts('relationship'))
    race = st.selectbox("Race", opts('race'))
    sex = st.selectbox("Sex", opts('sex'))

with col3:
    capital_gain = st.number_input("Capital Gain (USD)", 0, 100000, 2174)
    capital_loss = st.number_input("Capital Loss (USD)", 0, 100000, 0)
    hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", opts('native-country'),
                                  index=opts('native-country').index('United-States')
                                  if 'United-States' in opts('native-country') else 0)

# 3. Prediction
if st.button("Predict Income Class", type="primary"):
    # Build a single-row DataFrame with the RAW values (same shape as training X)
    raw_row = pd.DataFrame([{
        'age': age,
        'education-num': education_num,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'native-country': native_country,
    }])

    try:
        # One-hot encode the same way as training
        encoded_row = pd.get_dummies(raw_row, columns=categorical_cols, drop_first=True)

        # Align to training columns: add any missing dummy cols as 0, drop extras, reorder
        encoded_row = encoded_row.reindex(columns=feature_columns, fill_value=0)

        # Scale
        scaled = scaler.transform(encoded_row)
        scaled_df = pd.DataFrame(scaled, columns=feature_columns)

        # Predict
        prediction = model.predict(scaled_df)
        pred_label = target_encoder.inverse_transform(prediction)[0]

        # Probabilities
        prob_text = ""
        prob_map = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_df)[0]
            classes = target_encoder.inverse_transform(model.classes_)
            prob_map = dict(zip(classes, proba))
            prob_text = " | ".join([f"{cls}: {p*100:.1f}%" for cls, p in prob_map.items()])

        if str(pred_label).strip() == '>50K':
            st.success(f"### Predicted Income: **{pred_label}** 💵")
        else:
            st.info(f"### Predicted Income: **{pred_label}**")

        if prob_text:
            st.caption(f"Class probabilities — {prob_text}")
            high_income_prob = next((p for cls, p in prob_map.items()
                                     if str(cls).strip() == '>50K'), 0.0)
            st.progress(min(max(float(high_income_prob), 0.0), 1.0))

    except Exception as e:
        st.error(f"Prediction Error: {e}")