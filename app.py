import streamlit as st
import torch
import requests
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "best_saved.pt"

# ----------------------------------------------------------
# DOWNLOAD MODEL JIKA BELUM ADA
# ----------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive... Please wait.")
        r = requests.get(MODEL_DRIVE_URL, allow_redirects=True)
        open(MODEL_PATH, "wb").write(r.content)
        st.success("Model downloaded successfully!")

# ----------------------------------------------------------
# LOAD MODEL STREAMLIT CACHE
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    return YOLO(MODEL_PATH)

model = load_model()

st.title("♻️ Waste Object Detection – YOLO Realtime")

# ==========================================================
# MODE PILIHAN: WEBCAM / UPLOAD
# ==========================================================
mode = st.selectbox("Pilih mode:", ["Upload Foto", "Realtime Webcam"])

# ==========================================================
# MODE UPLOAD FOTO
# ==========================================================
if mode == "Upload Foto":
    uploaded = st.file_uploader("Upload gambar sampah:", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar asli", use_column_width=True)

        results = model(img)

        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Hasil Deteksi", use_column_width=True)

# ==========================================================
# MODE REALTIME WEBCAM
# ==========================================================
elif mode == "Realtime Webcam":
    st.write("Nyalakan webcam untuk deteksi realtime")

    run = st.checkbox("Start")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca webcam.")
            break

        results = model(frame)
        annotated = results[0].plot()

        FRAME_WINDOW.image(annotated, channels="BGR")

    cap.release()
