import streamlit as st
import cv2
import av
import supervision as sv
from inference_sdk import InferenceHTTPClient
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# ==================================================
# 1. KONFIGURASI
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Supervision", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# Konfigurasi STUN Server (Agar koneksi webcam stabil)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==================================================
# 2. CLASS PEMROSES VIDEO
# ==================================================
class WasteDetector:
    def __init__(self):
        # 1. Inisialisasi Client Roboflow (Hanya sekali di awal)
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )
        
        # 2. Setup Annotator (Supervision)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        # 3. Variabel Memori
        self.frame_count = 0
        self.skip_frames = 10 # Default skip (bisa diatur)
        self.last_detections = None # Menyimpan hasil deteksi terakhir

    def recv(self, frame):
        """
        Fungsi ini dipanggil berulang-ulang untuk setiap frame video
        """
        # Konversi format AV (WebRTC) ke OpenCV (Numpy)
        img = frame.to_ndarray(format="bgr24")
        
        # --- LOGIKA SKIP FRAME ---
        # Kita hanya kirim ke API jika sisa bagi frame_count == 0
        if self.frame_count % self.skip_frames == 0:
            try:
                # Kirim ke Roboflow API
                result = self.client.infer(img, model_id=MODEL_ID)
                
                # Konversi JSON ke format Supervision
                # result adalah dict, kita perlu pastikan formatnya aman
                self.last_detections = sv.Detections.from_inference(result)
                
            except Exception as e:
                print(f"Error API: {e}")
                # Jika error, biarkan last_detections tetap yang lama (jangan crash)

        # --- VISUALISASI ---
        # Gambar kotak menggunakan data TERAKHIR yang tersimpan di memori
        # Ini membuat kotak tetap muncul (persistent) meskipun kita sedang skip frame
        if self.last_detections:
            # Gambar Kotak
            img = self.box_annotator.annotate(scene=img, detections=self.last_detections)
            # Gambar Label
            img = self.label_annotator.annotate(scene=img, detections=self.last_detections)

        # Update counter
        self.frame_count += 1

        # Kembalikan gambar yang sudah dicoret-coret ke browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================================================
# 3. UI STREAMLIT
# ==================================================
st.title("♻️ Waste Detection (Supervision)")
st.write("Menggunakan Roboflow API + Supervision library.")

# Slider untuk mengatur Skip Frames (Biar user bisa atur kelancaran vs kuota)
skip_val = st.slider("Skip Frames (Makin tinggi makin hemat kuota & lancar)", 1, 30, 5)

def create_processor():
    processor = WasteDetector()
    processor.skip_frames = skip_val # Update nilai skip dari slider
    return processor

# Menjalankan Webcam
webrtc_streamer(
    key="waste-supervision",
    video_processor_factory=create_processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
