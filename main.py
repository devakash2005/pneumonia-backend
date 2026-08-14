import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import gc  # Added for memory management
from tensorflow.keras.applications.densenet import preprocess_input

# ---- Load model + metadata once, at startup ----
with open("metadata.json", "r") as f:
    metadata = json.load(f)

model = tf.keras.models.load_model("model_final.h5")

CLASS_NAMES = metadata["class_names"]          
IMG_SIZE = tuple(metadata["img_size"])          
THRESHOLD = metadata["decision_threshold"]      
UNCERTAIN_LOW = metadata["uncertain_band"]["low"]
UNCERTAIN_HIGH = metadata["uncertain_band"]["high"]
USE_TTA = metadata["use_tta_at_inference"]
TTA_N_AUG = metadata["tta_n_aug"]

app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*", 
        "http://localhost", 
        "https://localhost", 
        "capacitor://localhost"
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
    return arr


def predict_with_tta(img_array: np.ndarray, n_aug: int = 4) -> float:
    views = [img_array]
    views.append(np.fliplr(img_array))

    for _ in range(n_aug - len(views)):
        angle = np.random.uniform(-8, 8)
        rotated = tf.keras.preprocessing.image.apply_affine_transform(
            img_array, theta=angle, fill_mode="nearest"
        )
        views.append(rotated)

    batch = np.stack(views, axis=0)
    
    # OPTIMIZATION: Use model(..., training=False) instead of model.predict(...) to prevent memory leaks
    probs = model(batch, training=False).numpy().ravel()
    
    mean_prob = float(probs.mean())
    
    # MEMORY CLEANUP: Delete heavy arrays
    del views
    del batch
    del probs
    
    return mean_prob


# OPTIMIZATION: Allow Render's internal health check (HEAD request)
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running"}


def is_likely_xray(image_bytes: bytes) -> bool:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = img.resize((64, 64))
    arr = np.array(img_small).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()

    # MEMORY CLEANUP
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
        # Clear bytes from memory if rejected
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

    if USE_TTA:
        prob_pneumonia = predict_with_tta(img_array, n_aug=TTA_N_AUG)
    else:
        # OPTIMIZATION: model() instead of model.predict()
        expanded_array = np.expand_dims(img_array, axis=0)
        prob_pneumonia = float(model(expanded_array, training=False).numpy()[0][0])
        del expanded_array

    # MEMORY CLEANUP: Delete everything we don't need before returning the response
    del image_bytes
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