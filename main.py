import os

# Limit TensorFlow threading to minimize memory footprint
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import io
import gc
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.densenet import preprocess_input

# Restrict thread pools inside TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ---- Load model + metadata once ----
with open("metadata.json", "r") as f:
    metadata = json.load(f)

model = tf.keras.models.load_model("model_final.h5")

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])
THRESHOLD = metadata["decision_threshold"]
UNCERTAIN_LOW = metadata["uncertain_band"]["low"]
UNCERTAIN_HIGH = metadata["uncertain_band"]["high"]

app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost",
        "https://localhost",
        "capacitor://localhost",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = preprocess_input(arr)
    del img
    return arr


def predict_single(img_array: np.ndarray) -> float:
    """Predicts a single image without building multi-tensor batches."""
    inp = np.expand_dims(img_array, axis=0)
    pred = float(model(inp, training=False).numpy()[0][0])
    del inp
    return pred


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running"}


def is_likely_xray(image_bytes: bytes) -> bool:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = img.resize((64, 64))
    arr = np.array(img_small).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()

    del img
    del img_small
    del arr
    return channel_diff < 15


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    image_bytes = await file.read()

    if not is_likely_xray(image_bytes):
        del image_bytes
        gc.collect()
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
        del image_bytes
        gc.collect()
        raise HTTPException(status_code=400, detail="Could not read this image. Try a different file.")

    del image_bytes

    # Run lightweight single-tensor inference
    prob_pneumonia = predict_single(img_array)

    del img_array
    gc.collect()

    if UNCERTAIN_LOW <= prob_pneumonia <= UNCERTAIN_HIGH:
        label = "UNCERTAIN"
    elif prob_pneumonia >= THRESHOLD:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    borderline_margin = (THRESHOLD - UNCERTAIN_LOW) if THRESHOLD > UNCERTAIN_LOW else 0.05

    if label == "NORMAL" and prob_pneumonia >= (UNCERTAIN_LOW - borderline_margin):
        suggestion = "Normal, but confidence is moderate — a follow-up scan is advisable if symptoms continue."
    elif label == "NORMAL":
        suggestion = "Normal — no signs of pneumonia detected."
    elif label == "UNCERTAIN":
        suggestion = "Result unclear — please consult a doctor for a proper diagnosis."
    else:
        suggestion = "Signs of pneumonia detected — please consult a doctor."

    return {
        "label": label,
        "probability_pneumonia": round(prob_pneumonia, 4),
        "threshold_used": THRESHOLD,
        "suggestion": suggestion,
        "disclaimer": metadata.get("disclaimer", ""),
    }