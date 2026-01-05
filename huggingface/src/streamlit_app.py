# Husayn El Sharif

import json
import os
import numpy as np
import streamlit as st

import tensorflow as tf
import tf_keras as keras  # Import tf_keras for Keras 3 compatibility
import tensorflow_hub as hub

from huggingface_hub import hf_hub_download

# -----------------------------
# Optional: Helpful CLI message (local/dev)
# -----------------------------
port = os.getenv("STREAMLIT_SERVER_PORT", "7860")
print(f"\n✅ Open the app in your browser at: http://localhost:{port}\n", flush=True)

# -----------------------------
# Config
# -----------------------------
IMG_H = 456
IMG_W = 456

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES_PATH = os.path.join(APP_DIR, "class_names.json")

# Hugging Face model repo + filename
# Create a model repo like: helsharif/retinal-disease-efficientnet-tf
# Upload your model file as: best_model.keras
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "helsharif/retinal-disease-efficientnet-tf")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_model.keras")

# If you need private model access, set HF_TOKEN as a Space secret.
HF_TOKEN = os.getenv("HF_TOKEN", None)

# -----------------------------
# Helpers
# -----------------------------
def load_class_names(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


def preprocess_bytes_to_tensor(image_bytes: bytes, img_height=IMG_H, img_width=IMG_W) -> tf.Tensor:
    """
    Mirrors your notebook preprocessing:
    - decode image
    - pad to square
    - convert to float32 [0..1]
    - resize to (456,456)
    Returns: (H,W,3) float32 tensor
    """
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    side = tf.maximum(h, w)
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=(side - h) // 2,
        offset_width=(side - w) // 2,
        target_height=side,
        target_width=side,
    )

    image = tf.image.convert_image_dtype(image, tf.float32)  # [0..1]
    image = tf.image.resize(image, size=[img_height, img_width], method=tf.image.ResizeMethod.AREA)
    return image


def download_model_from_hub(repo_id: str, filename: str) -> str:
    """
    Downloads model artifact from Hugging Face Hub (cached on disk by huggingface_hub).
    Returns the local file path.
    """
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN,     # None for public repos
        repo_type="model",
    )
    return model_path


@st.cache_resource
def load_model_and_labels():
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)

    # Download model from HF Hub
    model_path = download_model_from_hub(MODEL_REPO_ID, MODEL_FILENAME)

    # Load model (KerasLayer needed if your model uses TF Hub layers)
    model = keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer},
        compile=False,  # fine for inference
    )
    return model, class_names, model_path


def predict_single_image_bytes(model, class_names, image_bytes: bytes):
    img = preprocess_bytes_to_tensor(image_bytes)
    x = tf.expand_dims(img, axis=0)  # (1,H,W,3)

    probs = model.predict(x, verbose=0)[0]
    probs = np.asarray(probs).astype(float)

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    topk_idx = probs.argsort()[::-1]
    topk = [(class_names[i], float(probs[i])) for i in topk_idx]

    return pred_label, confidence, probs, topk


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Eye Disease Classifier", page_icon="👁️", layout="centered")

st.title("👁️ Eye Disease Prediction")
st.write("Upload a single retinal image and get a model prediction.")
st.info(
    "📥 **Try the app with example retinal images**\n\n"
    "[Download example images from my GitHub Repository]"
    "(https://github.com/helsharif/end-to-end-retinal-disease-classification/tree/main/example_images)"
)

with st.expander("Model details", expanded=False):
    st.write("- Input size:", (IMG_H, IMG_W))
    st.write("- Classes loaded from:", CLASS_NAMES_PATH)
    st.write("- Model repo:", f"`{MODEL_REPO_ID}`")
    st.write("- Model file:", f"`{MODEL_FILENAME}`")

# Load model
try:
    model, class_names, model_path = load_model_and_labels()
    st.caption(f"Loaded model from HF Hub: `{MODEL_REPO_ID}/{MODEL_FILENAME}`")
except Exception as e:
    st.error("Failed to load model / labels.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image_bytes = uploaded.read()

    st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            pred_label, confidence, probs, topk = predict_single_image_bytes(model, class_names, image_bytes)

        st.subheader("Prediction")
        st.write(f"**Predicted class:** `{pred_label}`")
        st.write(f"**Confidence:** `{confidence:.3f}`")

        st.subheader("Class probabilities")
        chart_data = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)

        with st.expander("Top-k details"):
            for label, p in topk:
                st.write(f"- `{label}`: {p:.4f}")
else:
    st.info("Upload a JPG/PNG image to enable prediction.")


st.write("")  # spacer
st.write("")
# -----------------------------
# Footer (author credit)
# -----------------------------
st.markdown(
    "<hr style='margin-top:2rem;margin-bottom:1rem;'>"
    "<div style='text-align:center; font-size:0.85em; color:gray;'>"
    "Built by <b>Husayn El Sharif</b> · "
    "<a href='https://github.com/helsharif' target='_blank'>https://github.com/helsharif</a>"
    "</div>",
    unsafe_allow_html=True,
)
