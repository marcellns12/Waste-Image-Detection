import streamlit as st
import cv2
import av
import supervision as sv
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Live", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# ==================================================
# 2. CONFIG STUN SERVER (SOLUSI VIDEO GAK MUNCUL)
# ==================================================
# Kita pakai banyak server agar kalau satu diblokir, ada cadangan.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]}
)

# ==================================================
# 3. CLASS LOGIKA DETEKSI
# ==================================================
class WasteDetector:
    def __init__(self):
        # Setup API
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )
        # Setup Visual
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        # Variabel
        self.frame_count = 0
        self.skip_frames = 15  # Deteksi tiap 15 frame (biar video lancar)
        self.last_detections = None

    def recv(self, frame):
        # Convert ke gambar (NumPy)
        img = frame.to_ndarray(format="bgr24")
        
        # --- LOGIKA HEMAT KUOTA & ANTI LAG ---
        # Kirim ke API cuma sesekali, jangan setiap frame
        if self.frame_count % self.skip_frames == 0:
            try:
                # Resize kecil dulu biar upload cepat (opsional)
                # img_small = cv2.resize(img, (640, 640))
                
                # Kirim ke Roboflow
                result = self.client.infer(img, model_id=MODEL_ID)
                
                # Simpan hasil deteksi
                self.last_detections = sv.Detections.from_inference(result)
            except Exception:
                # Kalau internet error, jangan bikin video mati. Lanjut aja.
                pass

        # --- GAMBAR KOTAK (Persistent) ---
        # Kotak akan tetap digambar meskipun kita lagi nge-skip frame
        if self.last_detections:
            try:
                img = self.box_annotator.annotate(scene=img, detections=self.last_detections)
                img = self.label_annotator.annotate(scene=img, detections=self.last_detections)
            except Exception:
                pass

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================================================
# 4. TAMPILAN UTAMA
# ==================================================
st.title("♻️ Live Waste Detection")
st.info("Tunggu status berubah jadi 'Running'. Jika lama, refresh halaman.")

webrtc_streamer(
    key="waste-live",
    video_processor_factory=WasteDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
