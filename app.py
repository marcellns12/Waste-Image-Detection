import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import gdown
import io

st.set_page_config(page_title="Waste Detection Mobile", layout="wide")

# ===========================================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# ===========================================
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "waste_model.pt"

@st.cache_resource
def load_model():
    # kalau model belum ada → download
    try:
        with open(MODEL_PATH, "rb"):
            pass
    except:
        gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

    return YOLO(MODEL_PATH)

model = load_model()


# ===========================================
# KATEGORI + WARNA
# ===========================================
CATEGORY_COLOR = {
    "organic": (0, 255, 0),    # hijau
    "anorganic": (0, 255, 255),# kuning
    "b3": (0, 0, 255)          # merah
}

def get_color(label):
    return CATEGORY_COLOR.get(label.lower(), (255, 255, 255))


# ===========================================
# DRAW BOX
# ===========================================
def draw_boxes(image, results):
    img = np.array(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            label = r.names[cls].lower()
            conf = float(box.conf[0])

            color = get_color(label)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(
                img,
                f"{label.upper()} ({conf:.2f})",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
    return img


# ===========================================
# CUSTOM CAMERA FRONT/BACK
# ===========================================
st.markdown("""
<style>
#camera-container {text-align: center;}
video {width: 100%; border-radius: 10px;}
canvas {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("📱 Waste Detection – Mobile Camera")
st.write("Deteksi sampah **Organik (Hijau)**, **Anorganik (Kuning)**, dan **B3 (Merah)** secara real-time dari kamera HP kamu.")

camera_type = st.selectbox("Pilih Kamera", ["Belakang", "Depan"])
facing_mode = "environment" if camera_type == "Belakang" else "user"


# ===========================================
# CAMERA SCRIPT
# ===========================================
st.markdown(f"""
<div id="camera-container">
<video id="video" autoplay playsinline></video>
<button onclick="captureImage()">Capture</button>
<canvas id="canvas"></canvas>
</div>

<script>
let video = document.getElementById('video');

navigator.mediaDevices.getUserMedia({{
    video: {{ facingMode: "{facing_mode}" }}
}})
.then(stream => {{
    video.srcObject = stream;
}})
.catch(err => {{
    alert("Akses kamera ditolak atau error: " + err);
}});

function captureImage() {{
    let canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    let dataURL = canvas.toDataURL('image/jpeg');
    window.parent.postMessage(dataURL, "*");
}}
</script>
""", unsafe_allow_html=True)


# ===========================================
# TERIMA CAPTURE DARI JAVASCRIPT
# ===========================================
def js_listen():
    try:
        return st.experimental_get_websocket_message()
    except:
        return None

msg = js_listen()

if msg and isinstance(msg["data"], str) and msg["data"].startswith("data:image"):
    b64 = msg["data"].split(",")[1]
    img_bytes = base64.b64decode(b64)

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.subheader("📸 Hasil Capture")
    st.image(image, use_column_width=True)

    # Detection
    results = model(image)
    detected = draw_boxes(image, results)

    st.subheader("🔍 Hasil Deteksi Sampah")
    st.image(detected, use_column_width=True)
