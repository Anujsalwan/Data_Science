import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Pandas 4.0 introduced Pandas4Warning; suppress if present, ignore otherwise.
try:
    warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)  # type: ignore[attr-defined]
except Exception:
    pass

# 1. Helper Function to generate/load the Scaler and training columns from the CSV
@st.cache_resource
def prepare_assets():
    """Recreate the EXACT preprocessing pipeline from the notebook.

    The notebook flow (Anuj_Salwan.ipynb, cells 41-50) is:
      1. read_csv("income_evaluation.csv")        # raw CSV has leading-space column names
      2. df.columns = df.columns.str.strip()      # strip column names
      3. df = df.drop(columns=['fnlwgt'])         # drop fnlwgt
      4. df.columns = df.columns.str.strip()      # strip again (notebook does it twice)
      5. strip whitespace from object cell values
      6. df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
      7. pd.get_dummies(..., drop_first=True) on all object cols except 'income'
      8. StandardScaler fit on ALL numeric columns minus 'income' (incl. dummy bool cols)

    However: the saved model may have been trained on an older version of the
    preprocessing (e.g. before the strip / before fnlwgt was dropped). We trust
    the model — read its expected feature names and align the CSV-derived
    columns to that list exactly.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(script_dir, 'best_xgboost_classifier.pkl')

    # CSV may be named either way depending on the project; try common names.
    csv_candidates = ['income_evaluation.csv', 'adult.csv', 'adult_income.csv']
    csv_path = next((os.path.join(script_dir, n) for n in csv_candidates
                     if os.path.exists(os.path.join(script_dir, n))), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"None of {csv_candidates} found in {script_dir}"
        )

    # Load the trained classifier
    model = joblib.load(model_path)

    # Read what the model ACTUALLY expects (source of truth).
    expected_features = None
    try:
        booster = model.get_booster()
        if booster.feature_names is not None:
            expected_features = list(booster.feature_names)
    except Exception:
        pass
    if expected_features is None and hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

    # Detect whether the model expects leading spaces in column names.
    # If the raw CSV has " workclass" etc. and the notebook DIDN'T strip before
    # encoding, the dummy columns will look like " workclass_Private".
    leading_space_in_model = (
        expected_features is not None
        and any(f.startswith(" ") for f in expected_features)
    )
    keeps_fnlwgt = (
        expected_features is not None
        and any(f.strip() == "fnlwgt" for f in expected_features)
    )

    # ---------- Reproduce notebook preprocessing on the CSV ----------
    df = pd.read_csv(csv_path)

    # Capture raw column names BEFORE any stripping so we can use them verbatim
    # when the user enters input. Map: stripped_name -> raw_name (e.g. "workclass" -> " workclass")
    raw_name_map = {c.strip(): c for c in df.columns}

    # If the model has stripped names, strip the CSV's column names too.
    # Otherwise leave them untouched so dummies inherit the leading space.
    if not leading_space_in_model:
        df.columns = df.columns.str.strip()

    # Drop fnlwgt only if the model was trained without it.
    if not keeps_fnlwgt:
        # Account for either ' fnlwgt' or 'fnlwgt'
        for c in list(df.columns):
            if c.strip() == "fnlwgt":
                df = df.drop(columns=[c])

    # Strip whitespace from string cell values (notebook does this regardless)
    str_like = ["object", "string"]
    for col in df.select_dtypes(include=str_like).columns:
        df[col] = df[col].astype(str).str.strip()

    # Identify the income (target) column — could be 'income' or ' income'
    target_col = next(c for c in df.columns if c.strip() == "income")
    df[target_col] = df[target_col].map({"<=50K": 0, ">50K": 1})

    # One-hot encode all object columns except the target (drop_first=True)
    cat_cols = [c for c in df.select_dtypes(include=str_like).columns if c != target_col]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Build X (drop target)
    X_full = df.drop(columns=[target_col])

    # Notebook scales ALL numeric columns (numeric ints/floats only —
    # bool dummies are NOT picked up by select_dtypes(include=[np.number]) in
    # modern pandas, only by include=['number','bool']). To match notebook
    # exactly we use np.number which excludes bools.
    scale_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    scaler.fit(X_full[scale_cols])

    # If the model gave us its expected feature names, use them as the canonical
    # column order. Otherwise fall back to whatever we built.
    final_columns = expected_features if expected_features is not None else X_full.columns.tolist()

    return model, scaler, final_columns, scale_cols, leading_space_in_model, keeps_fnlwgt, raw_name_map

# Initialize Model, Scaler and feature columns
try:
    (
        model,
        scaler,
        feature_columns,
        NUMERIC_COLS,
        LEADING_SPACE,
        KEEPS_FNLWGT,
        RAW_NAME_MAP,
    ) = prepare_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure 'best_xgboost_classifier.pkl' and 'income_evaluation.csv' (or 'adult.csv') are next to the app file.")
    st.stop()

st.set_page_config(page_title="Income Prediction", layout="wide")
st.title("💰 Income Prediction App")
st.write("Data Set Used : https://www.kaggle.com/datasets/lodetomasi1995/income-classification/")
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
    # Build a dict keyed by STRIPPED column names — we'll remap to raw names below.
    user_values = {
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
    if KEEPS_FNLWGT:
        # fnlwgt is a sampling weight, not asked of users — use a typical value.
        user_values["fnlwgt"] = 178356  # approx median of training set

    # Remap to the EXACT column names from the raw CSV (e.g. " workclass" if the
    # CSV had a leading space and the model was trained with that schema).
    if LEADING_SPACE:
        raw_input = {RAW_NAME_MAP.get(k, k): v for k, v in user_values.items()}
    else:
        raw_input = user_values

    try:
        # 1. Create DataFrame from raw input
        df = pd.DataFrame([raw_input])

        # 2. Strip whitespace from string CELL values (matches training; we do
        #    NOT strip column names here when LEADING_SPACE is true, because
        #    we're intentionally preserving the leading-space schema).
        str_like = ["object", "string"]
        for col in df.select_dtypes(include=str_like).columns:
            df[col] = df[col].astype(str).str.strip()

        # 3. One-hot encode. Note: drop_first=True breaks for single-row input
        #    because each categorical column has only one unique value, so
        #    get_dummies + drop_first eliminates ALL dummies. We use
        #    drop_first=False here; reindex below will keep only the columns
        #    the model expects (which already reflect drop_first=True from training).
        cat_cols = df.select_dtypes(include=str_like).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        # 4. Align columns to the model's expected feature space
        #    (adds missing dummies as 0, drops extras, enforces column order)
        features_df = df.reindex(columns=feature_columns, fill_value=0)

        # 5. Scale numeric columns with the same scaler used in training
        cols_to_scale = [c for c in NUMERIC_COLS if c in features_df.columns]
        if cols_to_scale:
            features_df[cols_to_scale] = scaler.transform(features_df[cols_to_scale])

        # 6. Cast to numeric float — XGBoost dislikes bool dummies in some versions
        features_df = features_df.astype(float)

        # 7. Predict
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