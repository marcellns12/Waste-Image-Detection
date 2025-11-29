import streamlit as st
import gdown
from ultralytics import YOLO
from PIL import Image
import cv2
import os

MODEL_URL = "https://drive.google.com/uc?id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "waste_model.pt"

# ----------------------------------------------------------
# DOWNLOAD MODEL AMAN (ANTI-KORUPSI)  
# ----------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive (via gdown)...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# ----------------------------------------------------------
# LOAD MODEL 
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("♻️ Waste Detection YOLO Realtime")

mode = st.selectbox("Mode:", ["Upload Foto", "Real-time Webcam"])

# ----------------------------------------------------------
# UPLOAD IMAGE
# ----------------------------------------------------------
if mode == "Upload Foto":
    file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Gambar asli")

        results = model(img)
        annotated = results[0].plot()

        st.image(annotated, caption="Hasil Deteksi")

# ----------------------------------------------------------
# WEBCAM REALTIME
# ----------------------------------------------------------
else:
    run = st.checkbox("Start Webcam")
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam error.")
            break

        results = model(frame)
        annotated = results[0].plot()

        frame_window.image(annotated, channels="BGR")

    cap.release()
