import json
import os
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st


# -----------------------------
# Page setup (same look, two-panel layout)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification through Artificial Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Hidden paths (NO sidebar settings)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "image_classification_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()


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
def load_model_cached(model_path: str, mtime: float):
    # Cache the model so it doesn't reload on every Streamlit rerun
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError:
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
    """
    Convert PIL image -> NumPy batch (1, H, W, 3)
    Resize to model input size.
    Do NOT divide by 255 if model already contains Rescaling(1/255).
    """
    img = img.convert("RGB")

    in_shape = getattr(model, "input_shape", None)  # (None, 256, 256, 3)
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img)              # (H, W, 3)
    x = np.expand_dims(x, 0)       # (1, H, W, 3)
    x = x.astype("float32")

    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    """Ensure the output behaves like probabilities. If not, apply softmax."""
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

if not MODEL_PATH.exists():
    model_error = "Model file not found ❗"
else:
    model_error = None

if not CLASSES_PATH.exists():
    classes_error = "class_names.json file not found ❗"
else:
    classes_error = None

if model_error is None:
    try:
        model = load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as e:
        model_error = f"Model found, but failed to load ❌\n\n{e}"

if classes_error is None:
    try:
        class_names = load_class_names(CLASSES_PATH)
    except Exception as e:
        classes_error = f"class_names.json found, but failed to load ❌\n\n{e}"


# -----------------------------
# TWO "PAGES" (LEFT / RIGHT)
# -----------------------------
left, right = st.columns([1, 2], gap="large")


# -----------------------------
# LEFT: User Manual (collapsible)
# -----------------------------
with left:
    with st.expander("📘 User Manual", expanded=False):
        st.markdown(
            """
**How to take a good photo (important):**
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (**no blur**).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid strong **shadows**, **reflections**, and **filters**.
- Don’t crop too tightly — include the **full infected area**.

**How to use the app:**
1. Click **Take/Upload Photo** and upload a leaf image (**PNG / JPG / JPEG**).
2. Wait a moment for the prediction.
3. Read the **predicted class** and **confidence**.
            """
        )


# -----------------------------
# RIGHT: Title + Upload + Predict
# -----------------------------
with right:
    st.title("🌿 Plant Disease identification\nthrough Artificial Intelligence")
    st.caption(
        "Upload a plant leaf image and this app will identify the plant disease using your trained artificial intelligence "
        "(TensorFlow/Keras model)."
    )

    st.divider()

    # Show errors
    if model_error:
        st.error("Model is not loaded. Please contact the app owner.")
        st.caption(model_error)
        st.stop()

    if classes_error:
        st.error("Class names are not loaded. Please contact the app owner.")
        st.caption(classes_error)
        st.stop()

    # ✅ Label changed to "Take/Upload Photo" (button inside still shows "Browse files")
    uploaded = st.file_uploader("Take/Upload Photo", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    # Reset/Clear image under the image
    if st.button("Reset / Clear image"):
        st.session_state["last_hash"] = None
        st.session_state["last_pred"] = None
        st.session_state["last_probs"] = None
        st.rerun()

    x = preprocess(img, model)

    if st.session_state.get("last_hash") != img_hash or st.session_state.get("last_probs") is None:
        preds = model.predict(x, verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 2:
            preds = preds[0]

        probs = to_probabilities(preds)
        pred_id = int(np.argmax(probs))

        if pred_id >= len(class_names):
            st.error(
                f"Prediction index {pred_id} is outside class_names list (length {len(class_names)}). "
                "Fix: class_names.json must match the model output order."
            )
            st.stop()

        st.session_state["last_hash"] = img_hash
        st.session_state["last_probs"] = probs
        st.session_state["last_pred"] = pred_id

    probs = st.session_state["last_probs"]
    pred_id = int(st.session_state["last_pred"])
    pred_label = class_names[pred_id]
    confidence = float(probs[pred_id])

    st.success(f"✅ Predicted class: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.subheader("3) Top predictions")
    top_k = min(5, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]

    for rank, i in enumerate(top_idx, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo with a plain background.")
