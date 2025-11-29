import streamlit as st
import cv2
import numpy as np
import av
import threading
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI API & SERVER
# ==================================================
st.set_page_config(page_title="♻️ Live Waste Detection", layout="wide")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# Client Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Konfigurasi STUN Server (Agar webcam stabil & tidak error connection)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

# ==================================================
# 2. LOGIKA PEMROSESAN VIDEO (CLASS)
# ==================================================
# Kita pakai Class agar bisa menyimpan 'state' (hasil deteksi terakhir)
class VideoProcessor:
    def __init__(self):
        self.last_predictions = None # Menyimpan hasil deteksi terakhir
        self.is_processing = False   # Menandakan apakah API sedang sibuk
        self.lock = threading.Lock() # Pengaman thread

    def api_worker(self, img_bgr):
        """Fungsi ini berjalan di background thread untuk request API"""
        temp_path = ""
        try:
            # Simpan temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, img_bgr)
            
            # Request ke Roboflow
            result = CLIENT.infer(temp_path, model_id=MODEL_ID)
            
            # Update hasil deteksi terakhir secara aman
            with self.lock:
                self.last_predictions = result
                
        except Exception as e:
            print(f"API Error: {e}")
        finally:
            self.is_processing = False
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def draw_box(self, image, predictions):
        """Menggambar kotak berdasarkan data prediksi"""
        if not predictions or 'predictions' not in predictions:
            return image
            
        for box in predictions['predictions']:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            label = box['class']
            conf = box['confidence']

            # Koordinat
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Gambar
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label} {conf:.1%}"
            (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - 25), (x1 + t_w, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image

    def recv(self, frame):
        """Fungsi utama yang dipanggil setiap frame video"""
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Cek apakah thread API sedang kosong (tidak sibuk)
        if not self.is_processing:
            self.is_processing = True
            # Jalankan request API di thread terpisah (Parallel)
            # Agar video tidak nge-freeze menunggu internet
            t = threading.Thread(target=self.api_worker, args=(img.copy(),))
            t.start()

        # 2. Gambar kotak menggunakan hasil prediksi TERAKHIR yang tersedia
        with self.lock:
            if self.last_predictions:
                img = self.draw_box(img, self.last_predictions)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================================================
# 3. UI STREAMLIT
# ==================================================
st.title("♻️ Realtime Waste Detection (API)")
st.info("Arahkan benda ke kamera. Deteksi berjalan otomatis.")

# Menjalankan Webcam
webrtc_streamer(
    key="waste-realtime",
    video_processor_factory=VideoProcessor, # Gunakan Class logic di atas
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("""
---
**Catatan:**
- Karena menggunakan **API Cloud**, mungkin ada sedikit *delay* antara gerakan benda dan kotak hijau (tergantung kecepatan internet).
- Video akan tetap lancar, tapi kotak hijaunya akan 'menyusul' gerakan benda.
""")
