"""
Income Prediction Streamlit App
--------------------------------
Predicts whether a person's income is >50K or <=50K using the UCI Adult dataset.

Required files in the same directory:
    - best_<model_name>_classifier.pkl    (the trained classifier)
    - train_columns.pkl                   (list of X_train.columns from the notebook)
    - scaler.pkl                          (the StandardScaler fit on training data)

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Income Prediction",
    page_icon="💰",
    layout="centered",
)

st.title("💰 Income Prediction App")
st.caption(
    "Predicts whether annual income exceeds **$50K** based on demographic and "
    "employment data (UCI Adult dataset)."
)


# ---------------------------------------------------------------------------
# 2. Load model + preprocessing artifacts (cached)
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = [
    "best_gradient_boosting_model.pkl",
]


@st.cache_resource
def load_artifacts():
    """Load model, training column list, and scaler. Cached across reruns."""
    # find whichever classifier .pkl is present
    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if model_path is None:
        st.error(
            "No classifier .pkl found. Expected one of: "
            + ", ".join(MODEL_CANDIDATES)
        )
        st.stop()

    if not os.path.exists("train_columns.pkl"):
        st.error(
            "`train_columns.pkl` is missing. Re-run training and add:\n\n"
            "```python\njoblib.dump(list(X_train.columns), 'train_columns.pkl')\n```"
        )
        st.stop()

    if not os.path.exists("scaler.pkl"):
        st.error(
            "`scaler.pkl` is missing. Re-run training and add:\n\n"
            "```python\njoblib.dump(scaler, 'scaler.pkl')\n```"
        )
        st.stop()

    model = joblib.load(model_path)
    train_columns = joblib.load("train_columns.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, train_columns, scaler, model_path


model, TRAIN_COLUMNS, scaler, model_path = load_artifacts()

# Numeric columns that the scaler was fit on (must match training exactly).
# In the notebook fnlwgt was DROPPED before scaling, so it's not here.
NUMERIC_COLS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


# ---------------------------------------------------------------------------
# 3. Dropdown options (taken directly from the training CSV)
# ---------------------------------------------------------------------------
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
    "State-gov", "Without-pay", "Never-worked", "?",
]
EDUCATION_OPTIONS = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
    "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
    "Masters", "Prof-school", "Doctorate",
]
# education-num mapping (matches the dataset's encoding)
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


# ---------------------------------------------------------------------------
# 4. Preprocessing — must mirror the notebook EXACTLY
# ---------------------------------------------------------------------------
def preprocess_input(raw: dict) -> pd.DataFrame:
    """Turn a dict of raw form values into a model-ready feature row.

    Steps (must match the training notebook):
      1. Build a one-row DataFrame.
      2. Drop `fnlwgt` (the notebook drops it before encoding).
      3. Strip whitespace from string columns.
      4. One-hot encode categorical columns with drop_first=True.
      5. Reindex to TRAIN_COLUMNS — adds missing dummies as 0, drops extras,
         enforces the exact training column order.
      6. Scale the numeric columns with the SAME StandardScaler used in training.
    """
    df = pd.DataFrame([raw])

    # Step 2: drop fnlwgt if present
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    # Step 3: strip whitespace from object columns (training did this)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Step 4: one-hot encode (drop_first=True, same as training)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Step 5: align columns to training feature space
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    # Step 6: scale numeric columns (only those that exist in TRAIN_COLUMNS)
    cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


# ---------------------------------------------------------------------------
# 5. Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    st.write(f"**File:** `{model_path}`")
    st.write(f"**Type:** `{type(model).__name__}`")
    st.write(f"**# features expected:** {len(TRAIN_COLUMNS)}")
    with st.expander("Show feature list"):
        st.write(TRAIN_COLUMNS)


# ---------------------------------------------------------------------------
# 6. Input form
# ---------------------------------------------------------------------------
st.subheader("Enter the person's details")

with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=37, step=1)
        education = st.selectbox("Education", EDUCATION_OPTIONS, index=EDUCATION_OPTIONS.index("HS-grad"))
        workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS, index=0)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=0)
        hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40, step=1)
        capital_gain = st.number_input("Capital gain", min_value=0, max_value=99999, value=0, step=100)

    with col2:
        sex = st.selectbox("Sex", SEX_OPTIONS, index=0)
        race = st.selectbox("Race", RACE_OPTIONS, index=0)
        marital_status = st.selectbox("Marital status", MARITAL_OPTIONS, index=0)
        relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS, index=0)
        native_country = st.selectbox("Native country", COUNTRY_OPTIONS, index=0)
        capital_loss = st.number_input("Capital loss", min_value=0, max_value=4356, value=0, step=50)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# 7. Prediction
# ---------------------------------------------------------------------------
if submitted:
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
        X_new = preprocess_input(raw_input)
        pred = model.predict(X_new)[0]

        # Probability if the model supports it
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]

        st.divider()
        st.subheader("Prediction")

        if int(pred) == 1:
            st.success("### 💸 Income is likely **> $50K**")
        else:
            st.info("### 💵 Income is likely **≤ $50K**")

        if proba is not None:
            c1, c2 = st.columns(2)
            c1.metric("P(income ≤ $50K)", f"{proba[0]*100:.1f}%")
            c2.metric("P(income > $50K)", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

        with st.expander("Show raw input sent to the model"):
            st.json(raw_input)

        with st.expander("Show processed feature row (first 30 cols)"):
            st.dataframe(X_new.iloc[:, :30])

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)