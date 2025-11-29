import streamlit as st
import cv2
import numpy as np
import tempfile
import urllib.request
from ultralytics import YOLO
import os

st.title("♻️ Waste Detection YOLO")

# === LINK HUGGING FACE ===
MODEL_HF_URL = "https://huggingface.co/Marcellfevaveavav/YoloV11m/resolve/main/waste_model.pt"
MODEL_PATH = "waste_model.pt"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_HF_URL, MODEL_PATH)
            st.success("✅ Model berhasil di-download dari Hugging Face")
        except Exception as e:
            st.error(f"Gagal download model: {e}")
    return MODEL_PATH

@st.cache_resource
def load_model():
    path = download_model()
    return YOLO(path)

model = load_model()

st.subheader("🟢 Pilih Mode")
mode = st.radio("Mode Detection:", ["Realtime Webcam", "Upload Foto"])

# ============================
# REALTIME WEBCAM
# ============================
if mode == "Realtime Webcam":
    st.write("Nyalakan kamera lalu arahkan sampah ke kamera.")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam gagal dibuka. Izinkan akses kamera.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Tidak bisa membaca frame dari kamera.")
                    break

                results = model.predict(frame, imgsz=640, conf=0.5)
                annotated = results[0].plot()
                FRAME_WINDOW.image(annotated, channels="BGR")

    if cap:
        cap.release()

# ============================
# UPLOAD FOTO
# ============================
elif mode == "Upload Foto":
    uploaded = st.file_uploader("Upload foto", type=["jpg", "jpeg", "png"])
    if uploaded:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded.read())
        temp_path = temp.name

        img = cv2.imread(temp_path)
        results = model.predict(img, imgsz=640, conf=0.5)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR", caption="Hasil Deteksi")
