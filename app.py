import json
from pathlib import Path
import io
import hashlib

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -----------------------------
# ✅ Robust paths (works local + Streamlit Cloud)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / "models").resolve()


def resolve_path(p: str) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (BASE_DIR / pp).resolve()


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Plant Disease identification through Artificial Intelligence", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease identification through Artificial Intelligence")
st.caption("Upload a plant leaf image and this app will identify the plant disease using your trained artifitial intelligence (TensorFlow/Keras model).")

# -----------------------------
# Beginner-friendly explanation
# -----------------------------
with st.expander("📘 Beginner explanation (click to open)"):
    st.markdown(
        """
**What happens when you upload an image?**

1. **You upload** a JPG/PNG leaf photo.
2. The app **loads your saved model**.
3. The app **resizes** the image to the model input size (e.g., 256×256).
4. The app runs `model.predict(...)` to get probabilities.
5. The app shows the predicted class + confidence.
        """
    )

# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list, e.g. ['class1','class2',...]")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # Cache model, but refresh cache when file changes (mtime changes)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError:
        # Some Keras versions support safe_mode
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
    """PIL -> (1,H,W,3) float32, resized to model input. Avoid double-rescale if model already has Rescaling."""
    img = img.convert("RGB")
    in_shape = getattr(model, "input_shape", None)

    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img).astype("float32")
    x = np.expand_dims(x, 0)

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
# Sidebar settings
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Find all .keras models in models/ (repo)
available_models = []
if MODELS_DIR.exists():
    available_models = sorted([p.relative_to(BASE_DIR).as_posix() for p in MODELS_DIR.glob("*.keras")])

# Choose default model
preferred = "models/image_classification_model_linux.keras"
default_model_path = preferred if preferred in available_models else (available_models[0] if available_models else preferred)

# Model selector (prevents typos)
if available_models:
    model_path_txt = st.sidebar.selectbox(
        "Model path",
        options=available_models,
        index=available_models.index(default_model_path) if default_model_path in available_models else 0,
    )
else:
    model_path_txt = st.sidebar.text_input("Model path", value=default_model_path)

default_classes_path = "class_names.json"
classes_path_txt = st.sidebar.text_input("Class names file", value=default_classes_path)

model_path = resolve_path(model_path_txt)
classes_path = resolve_path(classes_path_txt)

st.sidebar.markdown(
    """
**Where should my model be?**

- Put your model inside the `models/` folder.
- Example: `models/image_classification_model_linux.keras`
    """
)

# Debug info (helps instantly on Streamlit Cloud)
st.sidebar.caption(f"Resolved model path: {model_path}")
st.sidebar.caption(f"Model exists: {model_path.exists()}")
if MODELS_DIR.exists():
    st.sidebar.caption(f"Found .keras files: {len(available_models)}")

# -----------------------------
# Load model + class names
# -----------------------------
model = None
class_names = None

if model_path.exists():
    try:
        model = load_model_cached(str(model_path), model_path.stat().st_mtime)
        st.sidebar.success("Model loaded ✅")
        st.sidebar.caption(f"Input shape: {getattr(model, 'input_shape', None)}")
    except Exception as e:
        st.sidebar.error("Model found, but failed to load ❌")
        st.sidebar.write(str(e))
else:
    st.sidebar.warning("Model file not found ❗")
    st.sidebar.caption("Select the correct model file from the dropdown above.")

if classes_path.exists():
    try:
        class_names = load_class_names(str(classes_path))
        st.sidebar.success(f"Loaded {len(class_names)} class names ✅")

        # Sanity check
        try:
            if model is not None and hasattr(model, "output_shape"):
                out_dim = model.output_shape[-1]
                if out_dim is not None and int(out_dim) != len(class_names):
                    st.sidebar.error(
                        f"Mismatch: model outputs {int(out_dim)} classes, but class_names.json has {len(class_names)}."
                    )
        except Exception:
            pass

    except Exception as e:
        st.sidebar.error("class_names.json found, but failed to load ❌")
        st.sidebar.write(str(e))
else:
    st.sidebar.warning("class_names file not found ❗")
    st.sidebar.caption("Keep class_names.json next to app.py (repo root) or change the path above.")

st.divider()

# -----------------------------
# Stop early if not ready
# -----------------------------
if model is None:
    st.error("Model is not loaded. Please fix the model path in the sidebar.")
    st.stop()

if class_names is None:
    st.error("class_names.json is not loaded. Please fix the class names file path in the sidebar.")
    st.stop()

# -----------------------------
# Upload + Predict
# -----------------------------
if st.sidebar.button("Reset / Clear image"):
    st.session_state["last_hash"] = None
    st.session_state["last_pred"] = None
    st.session_state["last_probs"] = None
    st.rerun()

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="uploader")

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

img_bytes = uploaded.getvalue()
img_hash = hashlib.md5(img_bytes).hexdigest()

img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

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
    if pred_id >= len(class_names):
        st.error(
            f"Prediction index {pred_id} is outside your class_names list (length {len(class_names)}).\n\n"
            "Fix: make sure `class_names.json` matches the model output order."
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

st.subheader("Top predictions")
top_k = min(5, len(probs))
top_idx = np.argsort(probs)[::-1][:top_k]
for rank, i in enumerate(top_idx, start=1):
    st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

st.caption("Tip: If predictions look wrong, double-check your class_names.json order and preprocessing.")
