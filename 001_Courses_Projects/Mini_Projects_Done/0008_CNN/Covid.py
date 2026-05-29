import os
import json
import numpy as np
import streamlit as st
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16    import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre

# --- Config: Google Drive file IDs ----------------------------------------
# Get these from the Drive SHARE link of each file (set to "Anyone with the link").
# A file link looks like:  https://drive.google.com/file/d/THIS_PART_IS_THE_ID/view
MODEL_FILE_ID = "1m642cFoayk6OC8B41O5RAfG_0hbjoWx9"   # best_model.keras  (the big one)

# Optional: if you also host the meta JSON on Drive, paste its ID here.
# Leave as None and just commit models/best_model_meta.json to your repo instead.
META_FILE_ID  = None                              # e.g. "1AbC...xyz" or None

MODEL_PATH = "models/best_model.keras"
META_PATH  = "models/best_model_meta.json"

# --- Page setup -----------------------------------------------------------
st.set_page_config(page_title="COVID-19 X-ray Classifier",
                   page_icon="X",
                   layout="centered")
st.title("COVID-19 Detection from Chest X-rays")
st.write("Upload a chest X-ray image and the model will predict whether it shows "
         "**COVID-19**, **Normal lungs**, or **Viral Pneumonia**.")

# --- Download helper ------------------------------------------------------
def download_from_drive(file_id, output):
    """Download a file from Google Drive only if it isn't already present."""
    if not os.path.exists(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# --- Load model + metadata (cached) ---------------------------------------
@st.cache_resource
def load_artifacts():
    # Download the model weights from Drive (runs once per container)
    download_from_drive(MODEL_FILE_ID, MODEL_PATH)

    # Meta JSON: download it too if an ID was given, otherwise expect it in the repo
    if META_FILE_ID:
        download_from_drive(META_FILE_ID, META_PATH)

    model = load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_artifacts()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

CLASSES    = meta["classes"]
IMG_SIZE   = meta["img_size"]
PREPROCESS = meta["preprocess"]
st.caption(f"Loaded model: **{meta['model_name']}** "
           f"(input {IMG_SIZE}x{IMG_SIZE}, preprocess: {PREPROCESS})")

# --- Image preprocessing --------------------------------------------------
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    if PREPROCESS == "rgb_norm":
        arr = arr / 255.0
    elif PREPROCESS == "vgg":
        arr = vgg_pre(arr)
    elif PREPROCESS == "resnet":
        arr = resnet_pre(arr)
    return np.expand_dims(arr, 0)

# --- Upload UI ------------------------------------------------------------
uploaded = st.file_uploader("Choose a chest X-ray image",
                            type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input X-ray", use_container_width=True)
    with col2:
        with st.spinner("Predicting..."):
            x = preprocess(img)
            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

        st.subheader("Prediction")
        st.success(f"**{CLASSES[pred_idx]}**  -  "
                   f"confidence {probs[pred_idx] * 100:.1f}%")

        st.write("Class probabilities:")
        for cls, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{cls}: {p * 100:.1f}%")

st.markdown("---")
st.caption("For educational use only. Not a medical device. Always consult a "
           "qualified radiologist for diagnosis.")
