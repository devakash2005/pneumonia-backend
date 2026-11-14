import os
import io
import gc
import json
import asyncio
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ai_edge_litert.interpreter import Interpreter

# Database imports
from database import engine, get_db
import models

# Modern Google GenAI SDK Import
from google import genai
from google.genai import types

# === SECURE API KEY CONFIGURATION ===
ai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---- Load metadata + lightweight model once ----
with open("metadata.json", "r") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])
THRESHOLD = metadata["decision_threshold"]
UNCERTAIN_LOW = metadata["uncertain_band"]["low"]
UNCERTAIN_HIGH = metadata["uncertain_band"]["high"]

# Same math as Keras DenseNet preprocess_input, written in plain NumPy
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

interpreter = Interpreter(model_path="model_final.tflite", num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Warm-up: do the first-time memory allocation at startup, not on the first upload
interpreter.set_tensor(input_details["index"], np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))
interpreter.invoke()
gc.collect()

app = FastAPI(title="Pneumonia Detection API")

# Automatically create PostgreSQL tables on startup
models.Base.metadata.create_all(bind=engine)

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
    img = Image.open(io.BytesIO(image_bytes))
    img.draft("RGB", (512, 512))  # JPEG only: decode at reduced size to save memory
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    del img
    return arr


def predict_single(img_array: np.ndarray) -> float:
    """Predicts a single image with the TFLite interpreter."""
    inp = np.expand_dims(img_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    pred = float(interpreter.get_tensor(output_details["index"])[0][0])
    del inp
    return pred


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running"}


def is_likely_xray(image_bytes: bytes) -> bool:
    img = Image.open(io.BytesIO(image_bytes))
    img.draft("RGB", (512, 512))  # JPEG only: decode at reduced size to save memory
    img = img.convert("RGB")
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

    try:
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

        # Run lightweight single-tensor inference
        prob_pneumonia = predict_single(img_array)

    finally:
        # Free RAM right away to stay under Render's memory limit
        if 'image_bytes' in locals():
            del image_bytes
        if 'img_array' in locals():
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


# ---- Gemini-Powered Virtual Doctor Assistant Endpoint (Strictly Scoped) ----
class DoctorQuery(BaseModel):
    prompt: str
    user_email: str = None  # Optional: logs conversation to DB if provided


# Models are tried in this order. If one is unavailable, the next is used.
# (Google retires old model names, so update this list if you see 404 NOT_FOUND in the logs.)
GEMINI_MODELS = ['gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite']

FRIENDLY_ERROR = (
    "Sorry, the assistant is not available right now. Please try again in a moment, "
    "or tap one of the quick questions. If you have trouble breathing or chest pain, "
    "please see a doctor or go to the nearest hospital."
)


@app.post("/virtual-doctor")
async def virtual_doctor_response(query: DoctorQuery, db: Session = Depends(get_db)):
    def call_gemini(model_name):
        return ai_client.models.generate_content(
            model=model_name,
            contents=query.prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the virtual medical assistant for PneumoVision AI, a mobile app that detects pneumonia from X-ray scans. "
                    "Provide helpful, concise, and empathetic educational responses about respiratory health, symptoms, prevention methods, and medical triage next steps. "
                    "STRICT RULE: If a user asks about anything unrelated to pneumonia, respiratory health, or medical triage "
                    "(such as coding, python, general history, math, cooking, or unrelated diseases), you MUST refuse to answer "
                    "and reply strictly with: 'I am specialized exclusively in pneumonia and respiratory health triage. "
                    "How can I assist you with pneumonia symptoms, prevention, or medical next steps today?' "
                    "Always maintain professional boundaries and include a brief reminder that you are an AI assistant and they should consult a certified physician for medical diagnosis."
                ),
                max_output_tokens=2048,
                temperature=0.3,
            ),
        )

    reply = ""

    for model_id in GEMINI_MODELS:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(call_gemini, model_id), timeout=15.0
            )
            if response and getattr(response, 'text', None):
                reply = response.text
                break
        except Exception as e:
            # Technical details go to the Render logs only, never to the user
            print(f"Model {model_id} failed with error: {e}")
            continue

    if not reply:
        reply = FRIENDLY_ERROR

    # Automatically save chat history to PostgreSQL if user_email is sent from frontend
    if query.user_email:
        try:
            user_chat = models.ChatHistory(user_email=query.user_email, sender="user", message=query.prompt)
            ai_chat = models.ChatHistory(user_email=query.user_email, sender="ai", message=reply)
            db.add(user_chat)
            db.add(ai_chat)
            db.commit()
        except Exception as ex:
            print(f"Database chat save error: {ex}")

    return {"reply": reply}


# ---- PostgreSQL User, Scan History & Chat Endpoints ----

class UserProfile(BaseModel):
    email: str
    name: str
    photo: str = None

class ScanData(BaseModel):
    user_email: str
    label: str
    probability: float
    date: str

class ChatMessage(BaseModel):
    user_email: str
    sender: str  # 'user' or 'ai'
    message: str

@app.post("/api/user")
def save_user(profile: UserProfile, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == profile.email).first()
    if not user:
        user = models.User(email=profile.email, name=profile.name, photo=profile.photo)
        db.add(user)
    else:
        user.name = profile.name
        user.photo = profile.photo
    db.commit()
    db.refresh(user)
    return {"message": "User saved successfully", "email": user.email}

@app.post("/api/scans")
def save_scan(scan: ScanData, db: Session = Depends(get_db)):
    db_scan = models.ScanRecord(
        user_email=scan.user_email,
        label=scan.label,
        probability=scan.probability,
        date=scan.date
    )
    db.add(db_scan)
    db.commit()
    return {"message": "Scan recorded successfully"}

@app.get("/api/scans/{user_email}")
def get_user_scans(user_email: str, db: Session = Depends(get_db)):
    scans = db.query(models.ScanRecord).filter(models.ScanRecord.user_email == user_email).all()
    return scans

@app.post("/api/chat")
def save_chat_message(chat: ChatMessage, db: Session = Depends(get_db)):
    db_chat = models.ChatHistory(
        user_email=chat.user_email,
        sender=chat.sender,
        message=chat.message
    )
    db.add(db_chat)
    db.commit()
    return {"message": "Chat saved successfully"}

@app.get("/api/chat/{user_email}")
def get_user_chat_history(user_email: str, db: Session = Depends(get_db)):
    chats = db.query(models.ChatHistory).filter(models.ChatHistory.user_email == user_email).order_by(models.ChatHistory.timestamp.asc()).all()
    return chats