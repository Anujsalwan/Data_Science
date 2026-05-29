"""
Dry Bean Type Classifier — Streamlit App
-----------------------------------------
Run locally (no Docker required):

    pip install -r requirements.txt
    streamlit run streamlit_app.py

The app expects these artifacts (produced by Beans_Multiclass_Classification.ipynb)
to live in the same folder:
    - bean_model.pkl
    - scaler.pkl
    - label_encoder.pkl
    - feature_columns.pkl
    - feature_ranges.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dry Bean Type Classifier",
    page_icon="🫘",
    layout="wide",
)

st.title("🫘 Dry Bean Type Classifier")
st.caption(
    "Predicts the variety of a dry bean (Seker, Barbunya, Bombay, Cali, "
    "Dermason, Horoz, Sira) from 16 vision-derived shape and size features."
)

# ---------------------------------------------------------------------------
# Load artifacts (cached so they load only once per session)
# ---------------------------------------------------------------------------
ARTIFACTS = ["bean_model.pkl", "scaler.pkl", "label_encoder.pkl",
             "feature_columns.pkl", "feature_ranges.pkl"]


@st.cache_resource
def load_artifacts():
    missing = [f for f in ARTIFACTS if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, missing
    model    = joblib.load("001_Courses_Projects/Mini_Projects_Done/0007_Supervised_ML_Classifications/bean_model.pkl")
    scaler   = joblib.load("001_Courses_Projects/Mini_Projects_Done/0007_Supervised_ML_Classifications/scaler.pkl")
    le       = joblib.load("001_Courses_Projects/Mini_Projects_Done/0007_Supervised_ML_Classifications/label_encoder.pkl")
    cols     = joblib.load("001_Courses_Projects/Mini_Projects_Done/0007_Supervised_ML_Classifications/feature_columns.pkl")
    ranges   = joblib.load("001_Courses_Projects/Mini_Projects_Done/0007_Supervised_ML_Classifications/feature_ranges.pkl")
    return model, scaler, le, cols, ranges, []


model, scaler, le, feature_columns, feature_ranges, missing = load_artifacts()

if missing:
    st.error(
        "Missing artifact file(s): " + ", ".join(missing) +
        ". Please run **Beans_Multiclass_Classification.ipynb** first to generate them."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — input mode
# ---------------------------------------------------------------------------
st.sidebar.header("Input mode")
mode = st.sidebar.radio(
    "How do you want to provide bean measurements?",
    ["Manual sliders", "Upload CSV"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.write(
    "Best model: tuned **Random Forest** trained on the UCI Dry Bean dataset "
    "(13,611 samples × 16 features × 7 classes)."
)

# ---------------------------------------------------------------------------
# Class info — short description shown after prediction
# ---------------------------------------------------------------------------
CLASS_INFO = {
    "SEKER":    "Small, round, white bean. Common in Turkish cuisine.",
    "BARBUNYA": "Speckled cranberry bean. Larger, kidney-shaped.",
    "BOMBAY":   "The largest variety in this dataset — distinctive size.",
    "CALI":     "White kidney-shape bean, medium-large.",
    "HOROZ":    "Long, slender bean often used in stews.",
    "SIRA":     "Medium-sized, white-pink, oval shape.",
    "DERMASON": "The most common variety here — small, oval, white.",
}


def predict(values_df: pd.DataFrame):
    """Scale the input and run the model. Returns (label, probabilities Series)."""
    values_df = values_df[feature_columns]  # enforce column order
    scaled = scaler.transform(values_df)
    pred = model.predict(scaled)
    label = le.inverse_transform(pred)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled)[0]
        proba_series = pd.Series(probs, index=le.classes_).sort_values(ascending=False)
    else:
        proba_series = None
    return label, proba_series


# ---------------------------------------------------------------------------
# Mode 1: Manual sliders
# ---------------------------------------------------------------------------
if mode == "Manual sliders":
    st.subheader("Enter bean measurements")
    st.write(
        "Adjust the 16 features below. Defaults are set to the dataset mean. "
        "If you don't know a value, leaving it at the default is fine."
    )

    cols_per_row = 4
    user_values = {}
    for i in range(0, len(feature_columns), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col_widget, feat in zip(row_cols, feature_columns[i:i + cols_per_row]):
            r = feature_ranges[feat]
            # widen bounds slightly so the slider doesn't clamp at observed extremes
            span = r["max"] - r["min"]
            lo, hi = r["min"] - 0.05 * span, r["max"] + 0.05 * span
            # Use a number_input for tiny-range features (ShapeFactor2 etc.) for precision
            if span < 1:
                user_values[feat] = col_widget.number_input(
                    feat, value=float(r["mean"]),
                    min_value=float(lo), max_value=float(hi),
                    format="%.6f", step=(span / 100 or 0.0001),
                )
            else:
                user_values[feat] = col_widget.slider(
                    feat,
                    min_value=float(lo), max_value=float(hi),
                    value=float(r["mean"]),
                )

    st.markdown("---")
    if st.button("🔮 Predict bean type", type="primary"):
        input_df = pd.DataFrame([user_values])
        label, proba = predict(input_df)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.success(f"### Predicted class: **{label}**")
            st.write(CLASS_INFO.get(label, ""))
        with col_right:
            if proba is not None:
                st.write("**Class probabilities:**")
                st.bar_chart(proba)

        with st.expander("Show input values"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ---------------------------------------------------------------------------
# Mode 2: CSV upload (batch prediction)
# ---------------------------------------------------------------------------
else:
    st.subheader("Upload a CSV for batch prediction")
    st.write(
        "Your CSV must contain these columns (any extras are ignored):"
    )
    st.code(", ".join(feature_columns), language="text")

    # Offer a downloadable template
    template = pd.DataFrame(
        [{c: feature_ranges[c]["mean"] for c in feature_columns}]
    )
    st.download_button(
        "⬇️ Download CSV template",
        data=template.to_csv(index=False).encode(),
        file_name="bean_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing_cols = [c for c in feature_columns if c not in df_in.columns]
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.stop()

        scaled = scaler.transform(df_in[feature_columns])
        preds = model.predict(scaled)
        labels = le.inverse_transform(preds)

        out = df_in.copy()
        out["Predicted_Class"] = labels
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)
            out["Confidence"] = probs.max(axis=1).round(4)

        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(50))
        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode(),
            file_name="bean_predictions.csv",
            mime="text/csv",
        )

        st.write("**Predicted class distribution:**")
        st.bar_chart(pd.Series(labels).value_counts())
