import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# UI tweaks (left panel red + rename uploader button)
# -----------------------------
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
    background: #b00020;
    padding: 1.25rem 1rem;
    border-radius: 14px;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child * {
    color: #ffffff !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child summary {
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}
div[data-testid="stFileUploader"] button { font-size: 0px !important; }
div[data-testid="stFileUploader"] button::after {
    content: "Take/Upload Photo";
    font-size: 14px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "image_classification_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()

# -----------------------------
# Rules / thresholds
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

BRIGHTNESS_MIN = 0.12
BLUR_VAR_MIN = 60.0

# "green seed" must have at least this much area to trust it for bbox
GREEN_SEED_MIN_RATIO = 0.006  # ~0.6% of pixels

# exclude very dark pixels from being treated as plant (black mulch)
DARK_V_CUTOFF = 0.10

# bbox padding (increase a bit for multi-leaf / branches)
BBOX_PAD_FRAC = 0.12


# -----------------------------
# Loading helpers
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    def _has(layer) -> bool:
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False
    return _has(model)


# -----------------------------
# Pre/post-processing
# -----------------------------
def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    img = img.convert("RGB")

    in_shape = getattr(model, "input_shape", None)
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, 0)

    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    pred_vector = np.asarray(pred_vector, dtype=np.float32)
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or pred_vector.min() < 0.0 or pred_vector.max() > 1.0:
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


def image_quality(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    brightness = float(arr.mean() / 255.0)

    gray = arr.mean(axis=2)
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    lap = (up + down + left + right) - 4.0 * gray
    blur_var = float(lap.var())

    return {"brightness": brightness, "blur_var": blur_var}


# -----------------------------
# Mask utilities (no OpenCV)
# -----------------------------
def rgb_to_hsv(rgb01: np.ndarray):
    """rgb01: float array in [0,1], shape (H,W,3) -> h,s,v in [0,1]"""
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]

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
    return h, s, v


def dilate(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """Simple dilation using 4-neighborhood via rolls."""
    mask = mask.astype(bool)
    for _ in range(iters):
        nb = (
            mask
            | np.roll(mask, 1, 0) | np.roll(mask, -1, 0)
            | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
        )
        mask = nb
    return mask


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def pad_bbox(bbox, W, H, pad_frac: float = 0.12):
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad = int(pad_frac * max(bw, bh))

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1


# -----------------------------
# Plant "seed" mask (vegetation-first)
# -----------------------------
def green_seed_mask(img: Image.Image):
    """
    Create a vegetation-ish seed mask (good for multi-leaf / branch scenes).
    IMPORTANT: This is only used to compute bbox. Prediction uses ORIGINAL crop.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, s, v = rgb_to_hsv(arr)

    # vegetation hues (yellow->green), relaxed
    hsv_leaf = (h >= 0.08) & (h <= 0.60) & (s >= 0.10) & (v >= 0.12)

    # green dominance (helps when hue is noisy)
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    green_dom = (g > r * 1.03) & (g > b * 1.03) & (v >= 0.10)

    seed = hsv_leaf | green_dom

    # remove very dark pixels so black mulch is not selected
    seed = seed & (v >= DARK_V_CUTOFF)

    return seed, float(seed.mean())


# -----------------------------
# Corner-based fallback (but with dark-exclusion)
# -----------------------------
def mask_by_corners_bbox(img: Image.Image, patch: int = 24, percentile: float = 99.5, extra_margin: float = 0.05):
    """
    BBox-only fallback using corner background distance.
    We still exclude very dark pixels to avoid black mulch.
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    H, W, _ = arr.shape

    p = int(min(patch, H // 3, W // 3))
    if p < 4:
        return None

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

    raw = dist > thr

    # clean
    m = Image.fromarray((raw.astype(np.uint8) * 255), mode="L")
    m = m.filter(ImageFilter.MedianFilter(size=5))
    m = m.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask = (np.array(m) > 40)

    # exclude very dark pixels (prevents black plastic mulch being foreground)
    _, _, v = rgb_to_hsv(arr)
    mask = mask & (v >= DARK_V_CUTOFF)

    bbox = bbox_from_mask(mask)
    return bbox


# -----------------------------
# Main: crop selection (NO symptom loss)
# -----------------------------
def get_prediction_crop(img: Image.Image):
    """
    Returns:
      pred_img: ORIGINAL cropped image (used for prediction)
      info: dict with method + bbox
    """
    W, H = img.size

    # 1) vegetation-first bbox (best for multiple leaves/branches)
    seed, ratio = green_seed_mask(img)
    if ratio >= GREEN_SEED_MIN_RATIO:
        seed2 = dilate(seed, iters=3)  # include near-branch pixels
        bbox = bbox_from_mask(seed2)
        if bbox is not None:
            bbox = pad_bbox(bbox, W, H, pad_frac=BBOX_PAD_FRAC)
            x0, y0, x1, y1 = bbox
            return img.crop((x0, y0, x1 + 1, y1 + 1)), {"method": "green_seed", "bbox": bbox, "ratio": ratio}

    # 2) corner-based fallback bbox (but avoid dark pixels)
    bbox = mask_by_corners_bbox(img)
    if bbox is not None:
        bbox = pad_bbox(bbox, W, H, pad_frac=BBOX_PAD_FRAC)
        x0, y0, x1, y1 = bbox
        return img.crop((x0, y0, x1 + 1, y1 + 1)), {"method": "corners_bbox", "bbox": bbox, "ratio": ratio}

    # 3) final fallback: use full image
    return img, {"method": "none", "bbox": None, "ratio": ratio}


# -----------------------------
# Load model + class names
# -----------------------------
model_error = None
classes_error = None

if not MODEL_PATH.exists():
    model_error = "Model file not found ❗ (Expected inside /models)"

if not CLASSES_PATH.exists():
    classes_error = "class_names.json file not found ❗"

model = None
class_names = None

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
# Layout
# -----------------------------
left, right = st.columns([1, 3], gap="large")

with left:
    with st.expander("📘 User Manual", expanded=False):
        st.markdown(
            """
**Tips for best results:**
- Bright natural light (avoid very dark photos)
- Keep leaf/plant in focus (no blur)
- Multiple leaves/branches are OK
- Plain background helps, but is NOT required

**Important:** The app crops the plant area but predicts on the **original crop**, so symptoms are not removed.
            """
        )

with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
  <div style="font-size:2.6rem; font-weight:700; line-height:1.08;">
    Plant Disease identification with AI 🌿
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a plant image and this app will identify the plant disease using our trained AI model (TensorFlow/Keras)."
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
            "Upload a photo and get the result.\n"
            "For best results, follow the User Manual on the left.\n"
            "For any issues, please contact the app owner."
        )
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    if st.button("Reset / Clear image"):
        for k in ["last_hash", "last_pred", "last_probs"]:
            st.session_state.pop(k, None)
        st.rerun()

    # -----------------------------
    # NEW: crop plant area (prediction uses ORIGINAL crop)
    # -----------------------------
    pred_img, crop_info = get_prediction_crop(img)

    with st.expander("🧪 Show crop used for prediction", expanded=False):
        st.image(pred_img, use_container_width=True)
        st.caption(f"Crop method: {crop_info['method']} | green_seed_ratio: {crop_info.get('ratio', 0.0):.2%}")

    # quality checks on pred_img
    q = image_quality(pred_img)
    if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    # -----------------------------
    # Predict
    # -----------------------------
    x = preprocess(pred_img, model)

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
    confidence = float(probs[pred_id])

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence. Try a clearer/brighter photo or move closer to the leaf.")
        st.stop()

    pred_label = class_names[pred_id]

    st.success(f"✅ Predicted class: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.subheader("3) Top predictions (≥ 50%)")
    idx_over = np.where(np.asarray(probs) >= CONFIDENCE_THRESHOLD)[0]
    idx_over = idx_over[np.argsort(np.asarray(probs)[idx_over])[::-1]]

    for rank, i in enumerate(idx_over, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo and move closer to the leaf.")
