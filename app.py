import av
import cv2
import gdown
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from PIL import Image

MODEL_URL = "https://drive.google.com/uc?id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "best_saved.pt"


# ----------------------------------------------------------
# DOWNLOAD MODEL AMAN
# ----------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")


# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    return YOLO(MODEL_PATH)


model = load_model()

st.title("♻️ Realtime Waste Detection (Front/Back HP Camera)")


# ----------------------------------------------------------
# VIDEO PROCESSOR
# ----------------------------------------------------------
class WasteVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)
        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ----------------------------------------------------------
# WEBRTC CONFIG (WAJIB UNTUK HP)
# ----------------------------------------------------------
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
}


# ----------------------------------------------------------
# START CAMERA (REALTIME + PERMISSION OTOMATIS)
# ----------------------------------------------------------
webrtc_streamer(
    key="waste-detection",
    mode="recvonly",
    video_processor_factory=WasteVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": "environment"  # 🔥 otomatis kamera belakang HP
        }
    },
    rtc_configuration=RTC_CONFIGURATION
)

st.info("Arahkan sampah ke kamera — hasil deteksi muncul langsung di layar.")
