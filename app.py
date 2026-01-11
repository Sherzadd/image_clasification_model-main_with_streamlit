import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -----------------------------
# ✅ Paths (hidden from users)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "image_classification_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.caption("Upload a leaf image and this app will predict the plant disease class using a trained TensorFlow/Keras model.")


# -----------------------------
# 📘 User Manual (for end users)
# -----------------------------
with st.expander("📘 User Manual (click to open)", expanded=True):
    st.markdown(
        """
### How to take a good photo (important)
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (no blur).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid strong **shadows**, **reflections**, and **filters**.
- Don’t crop too tightly — include the **full infected area**.

### How to use the app
1. Click **Take a photo** OR **Upload an image**.
2. Wait a second until the prediction appears.
3. Read the predicted class and confidence.
        """
    )


# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float) -> tf.keras.Model:
    # Cache model, refresh cache when file changes (mtime changes)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError:
        # some environments support safe_mode
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    """Return True if the model (even nested) contains a Rescaling layer."""
    def _has(layer) -> bool:
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False

    return _has(model)


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """PIL -> (1,H,W,3) float32; resize to model input; avoid double-scaling."""
    img = img.convert("RGB")

    in_shape = getattr(model, "input_shape", None)  # e.g. (None, 256, 256, 3)
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img).astype("float32")  # (H,W,3)
    x = np.expand_dims(x, 0)            # (1,H,W,3)

    # Only scale if model doesn't already include Rescaling(1/255)
    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    """Ensure output behaves like probabilities; apply softmax if needed."""
    pred_vector = np.asarray(pred_vector).astype("float32")
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


# -----------------------------
# Load model + class names (hidden)
# -----------------------------
model = None
class_names = None

# Friendly checks (no settings UI)
if not MODEL_PATH.exists():
    st.error("⚠️ Model file is missing on the server. Please contact the app owner.")
    st.stop()

if not CLASSES_PATH.exists():
    st.error("⚠️ Class labels file is missing on the server. Please contact the app owner.")
    st.stop()

try:
    model = load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception:
    st.error("⚠️ The model could not be loaded on the server. Please contact the app owner.")
    st.stop()

try:
    class_names = load_class_names(CLASSES_PATH)
except Exception:
    st.error("⚠️ The class labels could not be loaded on the server. Please contact the app owner.")
    st.stop()

# Optional sanity check (quiet)
try:
    if hasattr(model, "output_shape") and model.output_shape[-1] is not None:
        out_dim = int(model.output_shape[-1])
        if out_dim != len(class_names):
            st.error("⚠️ Internal configuration mismatch (model classes vs class_names). Please contact the app owner.")
            st.stop()
except Exception:
    pass


# -----------------------------
# Input (camera OR upload)
# -----------------------------
st.subheader("📷 Take a photo or upload an image")

col1, col2 = st.columns(2)
with col1:
    cam = st.camera_input("Take a photo")
with col2:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Reset (optional)
if st.button("Reset / Clear image"):
    st.session_state.clear()
    st.rerun()

# Choose input priority: camera > upload
file_obj = cam if cam is not None else uploaded
if file_obj is None:
    st.info("Please take a photo or upload an image to get a prediction.")
    st.stop()

img_bytes = file_obj.getvalue()
img_hash = hashlib.md5(img_bytes).hexdigest()

img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
st.image(img, caption=f"Uploaded image (id: {img_hash[:8]})", use_container_width=True)

# Preprocess + predict (cache per image)
x = preprocess(img, model)

if st.session_state.get("last_hash") != img_hash or st.session_state.get("last_probs") is None:
    preds = model.predict(x, verbose=0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.asarray(preds)
    if preds.ndim == 2:
        preds = preds[0]  # (n_classes,)

    probs = to_probabilities(preds)
    pred_id = int(np.argmax(probs))

    # Safe guard
    if pred_id >= len(class_names):
        st.error("⚠️ Prediction index is outside the class list. Please contact the app owner.")
        st.stop()

    st.session_state["last_hash"] = img_hash
    st.session_state["last_probs"] = probs
    st.session_state["last_pred"] = pred_id

# Show results
probs = st.session_state["last_probs"]
pred_id = int(st.session_state["last_pred"])
pred_label = class_names[pred_id]
confidence = float(probs[pred_id])

st.success(f"✅ Predicted class: **{pred_label}**")
st.write(f"Confidence: **{confidence:.2%}**")

st.subheader("Top predictions")
top_k = min(5, len(probs))
top_idx = np.argsort(probs)[::-1][:top_k]
for rank, i in enumerate(top_idx, start=1):
    st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

st.caption("If results look wrong, try a brighter/sharper photo and ensure the leaf is clearly visible.")
