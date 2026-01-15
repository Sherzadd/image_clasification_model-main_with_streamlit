import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
import streamlit as st


# -----------------------------
# Page setup (same look, two-panel layout)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# UI tweaks (left panel red + title sizing + rename uploader button)
# -----------------------------
st.markdown(
    """
<style>
/* --- Make the LEFT panel red (the first column of the main horizontal block) --- */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
    background: #b00020;              /* red */
    padding: 1.25rem 1rem;
    border-radius: 14px;
}

/* Make text inside the left panel white for readability */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child * {
    color: #ffffff !important;
}

/* Slightly nicer expander header in the left panel */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child summary {
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}

/* --- Rename the "Browse files" button text inside the uploader --- */
div[data-testid="stFileUploader"] button {
    font-size: 0px !important;        /* hide original "Browse files" */
}
div[data-testid="stFileUploader"] button::after {
    content: "Take/Upload Photo";     /* new label */
    font-size: 14px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Hidden paths (NO sidebar settings)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "image_classification_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()

# -----------------------------
# Your rules (thresholds)
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

# Existing checks
LEAF_RATIO_MIN = 0.05        # how much of the image looks like vegetation-ish colors
BRIGHTNESS_MIN = 0.12        # too dark -> reject
BLUR_VAR_MIN = 60.0          # too blurry -> reject (adjust if needed)

# NEW: background masking sanity check (less strict than before)
KEPT_RATIO_MIN = 0.02        # was 0.05


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
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


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
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0) or (pred_vector.max() > 1.0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


def image_quality_and_leafness(img: Image.Image) -> dict:
    """
    Fast heuristics to reject obvious non-leaf images:
    - brightness (too dark)
    - blur (Laplacian variance)
    - leaf_ratio: % pixels in vegetation-ish hue range (HSV)
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    brightness = float(arr.mean() / 255.0)

    # Blur score: Laplacian variance (simple 4-neighbor Laplacian)
    gray = arr.mean(axis=2)
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    lap = (up + down + left + right) - 4.0 * gray
    blur_var = float(lap.var())

    # Leaf ratio via HSV (vectorized)
    rgb = arr / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 1e-6

    idx = mask & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = mask & (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = mask & (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0) % 1.0  # 0..1

    s = np.zeros_like(maxc)
    idx2 = maxc > 1e-6
    s[idx2] = delta[idx2] / maxc[idx2]
    v = maxc

    # vegetation-ish hues: yellow->green (tolerant)
    leaf_mask = (h >= 0.12) & (h <= 0.50) & (s >= 0.15) & (v >= 0.15)
    leaf_ratio = float(leaf_mask.mean())

    return {"brightness": brightness, "blur_var": blur_var, "leaf_ratio": leaf_ratio}


def mask_background_by_corners(
    img: Image.Image,
    patch: int = 24,
    percentile: float = 99.5,
    extra_margin: float = 0.05,
):
    """
    Mask background using 4 corner patches (works if corners are real background).
    Returns: masked_img, masked_cropped_img, info_dict
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    H, W, _ = arr.shape

    p = int(min(patch, H // 3, W // 3))
    if p < 4:
        return img, img, {"kept_ratio": 1.0, "threshold": 0.0}

    corners = np.concatenate([
        arr[:p, :p, :].reshape(-1, 3),
        arr[:p, W - p:, :].reshape(-1, 3),
        arr[H - p:, :p, :].reshape(-1, 3),
        arr[H - p:, W - p:, :].reshape(-1, 3),
    ], axis=0)

    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)

    corner_dist = np.linalg.norm(corners - bg[None, :], axis=1)
    thr = float(np.percentile(corner_dist, percentile) + extra_margin)

    mask = dist > thr

    m = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    m = m.filter(ImageFilter.MedianFilter(size=5))
    m = m.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask_c = (np.array(m) > 40)

    kept_ratio = float(mask_c.mean())

    mask_img = Image.fromarray((mask_c.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", img.size, (255, 255, 255))
    masked = Image.composite(img, white, mask_img)

    ys, xs = np.where(mask_c)
    masked_cropped = masked
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = int(0.06 * max((x1 - x0 + 1), (y1 - y0 + 1)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W - 1, x1 + pad)
        y1 = min(H - 1, y1 + pad)
        masked_cropped = masked.crop((x0, y0, x1 + 1, y1 + 1))

    return masked, masked_cropped, {"kept_ratio": kept_ratio, "method": "corners"}


def mask_background_hsv_leaf(img: Image.Image):
    """
    Fallback for non-plain backgrounds when the leaf is green-ish.
    Returns: masked_img, masked_cropped_img, info_dict
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # RGB -> HSV (vectorized)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    m = delta > 1e-6

    idx = m & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = m & (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = m & (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(maxc)
    idx2 = maxc > 1e-6
    s[idx2] = delta[idx2] / maxc[idx2]
    v = maxc

    # relaxed vegetation-ish thresholds
    hsv_leaf = (h >= 0.08) & (h <= 0.55) & (s >= 0.12) & (v >= 0.12)
    green_dom = (g > r * 1.03) & (g > b * 1.03) & (v >= 0.10)

    mask = hsv_leaf | green_dom

    m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    m_img = m_img.filter(ImageFilter.MedianFilter(size=5))
    m_img = m_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask_c = (np.array(m_img) > 40)

    kept_ratio = float(mask_c.mean())

    mask_img = Image.fromarray((mask_c.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", img.size, (255, 255, 255))
    masked = Image.composite(img, white, mask_img)

    ys, xs = np.where(mask_c)
    masked_cropped = masked
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = int(0.06 * max((x1 - x0 + 1), (y1 - y0 + 1)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(img.size[0] - 1, x1 + pad)
        y1 = min(img.size[1] - 1, y1 + pad)
        masked_cropped = masked.crop((x0, y0, x1 + 1, y1 + 1))

    return masked, masked_cropped, {"kept_ratio": kept_ratio, "method": "hsv"}


def mask_leaf_for_prediction(img: Image.Image):
    """
    1) Try corner-based masking
    2) If fails, try HSV masking
    3) If fails, use original image (DO NOT BLOCK)
    """
    masked1, cropped1, info1 = mask_background_by_corners(img)
    if info1["kept_ratio"] >= KEPT_RATIO_MIN:
        return masked1, cropped1, info1

    masked2, cropped2, info2 = mask_background_hsv_leaf(img)
    if info2["kept_ratio"] >= KEPT_RATIO_MIN:
        return masked2, cropped2, info2

    return img, img, {"kept_ratio": max(info1["kept_ratio"], info2["kept_ratio"]), "method": "none"}


# -----------------------------
# Load model + class names (hidden)
# -----------------------------
model = None
class_names = None

model_error = None
classes_error = None

if not MODEL_PATH.exists():
    model_error = "Model file not found ❗"

if not CLASSES_PATH.exists():
    classes_error = "class_names.json file not found ❗"

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
left, right = st.columns([1, 3], gap="large")


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
2. The app will **try to mask the background** (best effort).
3. If masking isn’t reliable, it will still predict using the original image.
            """
        )


# -----------------------------
# RIGHT: Title + Upload + Predict
# -----------------------------
with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
  <div style="font-size:2.6rem; line-height:1;"></div>
  <div style="font-size:2.6rem; font-weight:700; line-height:1.08;">
    Plant Disease identification with AI 🌿
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a plant leaf image and this app will identify the plant disease using our trained artificial intelligence model "
        "(TensorFlow/Keras)."
    )

    st.divider()

    if model_error:
        st.error("Model is not loaded. Please contact the app owner.")
        st.caption(model_error)
        st.stop()

    if classes_error:
        st.error("Class names are not loaded. Please contact the app owner.")
        st.caption(classes_error)
        st.stop()

    uploaded = st.file_uploader("Take/Upload Photo", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded is None:
        st.info(
            "Upload photos and get the result.\n"
            "For best results, follow the User Manual on the left.\n"
            "For any issues, please contact the app owner at .sherzadzabihullah@yahoo.com"
        )
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    if st.button("Reset / Clear image"):
        st.session_state["last_hash"] = None
        st.session_state["last_pred"] = None
        st.session_state["last_probs"] = None
        st.session_state["last_is_confident"] = None
        st.rerun()

    # -----------------------------
    # Mask background (best effort) -> model uses masked_cropped
    # -----------------------------
    masked, masked_cropped, mask_info = mask_leaf_for_prediction(img)

    with st.expander("🧪 Show background-masked image (used for prediction)", expanded=False):
        st.image(masked_cropped, use_container_width=True)
        st.caption(f"Mask method: {mask_info['method']} | Kept ratio: {mask_info['kept_ratio']:.2%}")

    if mask_info["method"] == "none":
        st.info("Masking was not reliable for this photo — using the original image for prediction.")

    # -----------------------------
    # Run your checks on the image we will predict with
    # -----------------------------
    q = image_quality_and_leafness(masked_cropped)

    if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    # Leafness HSV-green check can fail for very brown leaves,
    # so only block if BOTH leaf_ratio is very low AND we couldn't mask anything useful.
    if q["leaf_ratio"] < LEAF_RATIO_MIN and mask_info["method"] == "none":
        st.warning("⚠️ This does not look like a plant leaf. Please upload a clear leaf photo and try again.")
        st.stop()

    # -----------------------------
    # Predict (use masked_cropped)
    # -----------------------------
    x = preprocess(masked_cropped, model)

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

        best_conf = float(probs[pred_id])
        is_confident = best_conf >= CONFIDENCE_THRESHOLD

        st.session_state["last_hash"] = img_hash
        st.session_state["last_probs"] = probs
        st.session_state["last_pred"] = pred_id
        st.session_state["last_is_confident"] = is_confident

    probs = st.session_state["last_probs"]
    pred_id = int(st.session_state["last_pred"])
    confidence = float(probs[pred_id])

    # YOUR RULE:
    # show prediction only if best >= 50%
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    pred_label = class_names[pred_id]

    st.success(f"✅ Predicted class: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.subheader("3) Top predictions (≥ 50%)")
    idx_over = np.where(np.asarray(probs) >= CONFIDENCE_THRESHOLD)[0]
    idx_over = idx_over[np.argsort(np.asarray(probs)[idx_over])[::-1]]

    for rank, i in enumerate(idx_over, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo with a plain background.")
