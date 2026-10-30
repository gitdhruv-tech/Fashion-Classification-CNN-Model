"""
Fashion Item Classifier - presentation demo
Run:  streamlit run app.py
"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from preprocess import CLASS_NAMES, preprocess

MODEL_PATH = "fashion_cnn.keras"
SAMPLE_DIR = "samples"
UPPER_BODY = {"T-shirt/top", "Shirt", "Pullover", "Coat"}   # the classes the model mixes up

st.set_page_config(page_title="Fashion Item Classifier", page_icon="👕", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from tensorflow import keras          # imported here so the page opens quickly
    return keras.models.load_model(MODEL_PATH)


st.title("👕 Fashion Item Classifier")
st.caption("CNN trained on Fashion-MNIST  |  93.9% test accuracy  |  macro F1 0.939")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Put it in the same folder as app.py.")
    st.stop()
model = load_model()

# ---------------- sidebar ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Auto-tag confidence threshold", 0.50, 0.99, 0.90, 0.01,
                              help="Below this, the item is sent to human review.")
invert_choice = st.sidebar.selectbox("Invert colours?", ["Auto", "Yes", "No"],
                                     help="The model expects a light item on a DARK background.")
crop = st.sidebar.checkbox("Crop and centre the item", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** works best on a single item on a plain background. "
                    "Cluttered photos or people wearing clothes will not work well.")

# ---------------- input ----------------
source = st.radio("Image source", ["Upload an image", "Pick a sample"], horizontal=True)
img = None

if source == "Upload an image":
    f = st.file_uploader("Upload a clothing image", type=["png", "jpg", "jpeg", "webp"])
    if f:
        img = Image.open(f)
else:
    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
    if not files:
        st.info("No 'samples' folder found. Run the export cell in the notebook to create it.")
    else:
        choice = st.selectbox("Sample image", [os.path.basename(p) for p in files])
        img = Image.open(os.path.join(SAMPLE_DIR, choice))

# ---------------- prediction ----------------
if img is not None:
    small = preprocess(img, invert_choice.lower(), crop)           # (28, 28) uint8
    x = (small.astype("float32") / 255.0)[None, ..., None]         # (1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]

    order = np.argsort(-probs)
    top, second = order[0], order[1]
    conf = float(probs[top])

    c1, c2 = st.columns(2)
    c1.image(img.convert("RGB"), caption="Your image", use_container_width=True)
    c2.image(Image.fromarray(small).resize((224, 224), Image.NEAREST),
             caption="What the model sees (28x28)", use_container_width=True)

    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.metric("Confidence", f"{conf:.1%}")

    if conf >= threshold:
        st.success(f"Auto-tagged as **{CLASS_NAMES[top]}** (confidence above {threshold:.0%}).")
    else:
        st.warning(f"Confidence is below {threshold:.0%}. **Send to human review.**")

    if (CLASS_NAMES[top] in UPPER_BODY and CLASS_NAMES[second] in UPPER_BODY
            and probs[second] > 0.15):
        st.info(f"Close call between **{CLASS_NAMES[top]}** and **{CLASS_NAMES[second]}**. "
                "These are the classes the model confuses most (see the confusion matrix).")

    st.markdown("**Top 3 predictions**")
    for i in order[:3]:
        st.write(f"{CLASS_NAMES[i]}: {probs[i]:.1%}")

    st.markdown("**All class probabilities**")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))
