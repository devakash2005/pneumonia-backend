import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from tensorflow.keras.applications.densenet import preprocess_input

# ---- Load model + metadata once, at startup ----
with open("metadata.json", "r") as f:
    metadata = json.load(f)

model = tf.keras.models.load_model("model_final.h5")

CLASS_NAMES = metadata["class_names"]          # ["NORMAL", "PNEUMONIA"]
IMG_SIZE = tuple(metadata["img_size"])          # (224, 224)
THRESHOLD = metadata["decision_threshold"]      # 0.3958
UNCERTAIN_LOW = metadata["uncertain_band"]["low"]
UNCERTAIN_HIGH = metadata["uncertain_band"]["high"]
USE_TTA = metadata["use_tta_at_inference"]
TTA_N_AUG = metadata["tta_n_aug"]

app = FastAPI(title="Pneumonia Detection API")

# Explicitly allow Capacitor Android Webview origins along with a general wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*", 
        "http://localhost", 
        "https://localhost", 
        "capacitor://localhost"
    ],
    allow_credentials=False, # Must be False when using "*" in FastAPI
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Load raw image bytes -> resized, preprocessed array ready for the model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = preprocess_input(arr)
    return arr


def predict_with_tta(img_array: np.ndarray, n_aug: int = 4) -> float:
    """Same TTA logic as the training notebook: original + flip + small rotations, averaged."""
    views = [img_array]
    views.append(np.fliplr(img_array))

    for _ in range(n_aug - len(views)):
        angle = np.random.uniform(-8, 8)
        rotated = tf.keras.preprocessing.image.apply_affine_transform(
            img_array, theta=angle, fill_mode="nearest"
        )
        views.append(rotated)

    batch = np.stack(views, axis=0)
    probs = model.predict(batch, verbose=0).ravel()
    return float(probs.mean())


@app.get("/")
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running"}


def is_likely_xray(image_bytes: bytes) -> bool:
    """Rough guardrail: chest X-rays are grayscale/near-grayscale. A photo with strong,
    varied color content (e.g. a cat photo) is very unlikely to be an X-ray. This is a
    simple heuristic, not a real classifier — it only catches obviously wrong uploads."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = img.resize((64, 64))
    arr = np.array(img_small).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # For a grayscale-like image, R, G, and B channels are nearly identical per pixel.
    channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()

    return channel_diff < 15  # threshold picked loosely; tune if it misfires


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    image_bytes = await file.read()

    if not is_likely_xray(image_bytes):
        return {
            "label": "INVALID_IMAGE",
            "probability_pneumonia": None,
            "threshold_used": THRESHOLD,
            "suggestion": "Please upload a valid chest X-ray image.",
            "disclaimer": metadata.get("disclaimer", ""),
        }

    try:
        img_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image. Try a different file.")

    if USE_TTA:
        prob_pneumonia = predict_with_tta(img_array, n_aug=TTA_N_AUG)
    else:
        prob_pneumonia = float(model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0])

    # Decide label using the uncertain band first, then the recall-first threshold.
    if UNCERTAIN_LOW <= prob_pneumonia <= UNCERTAIN_HIGH:
        label = "UNCERTAIN"
    elif prob_pneumonia >= THRESHOLD:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    # Confidence-tier message: reflects how sure the model is, not a diagnosis or cause.
    # "Borderline normal" = below threshold but not far from the uncertain band's low edge.
    borderline_margin = (THRESHOLD - UNCERTAIN_LOW) if THRESHOLD > UNCERTAIN_LOW else 0.05

    if label == "NORMAL" and prob_pneumonia >= (UNCERTAIN_LOW - borderline_margin):
        suggestion = "Normal, but confidence is moderate — a follow-up scan is advisable if symptoms continue."
    elif label == "NORMAL":
        suggestion = "Normal — no signs of pneumonia detected."
    elif label == "UNCERTAIN":
        suggestion = "Result unclear — please consult a doctor for a proper diagnosis."
    else:  # PNEUMONIA
        suggestion = "Signs of pneumonia detected — please consult a doctor."

    return {
        "label": label,
        "probability_pneumonia": round(prob_pneumonia, 4),
        "threshold_used": THRESHOLD,
        "suggestion": suggestion,
        "disclaimer": metadata.get("disclaimer", ""),
    }