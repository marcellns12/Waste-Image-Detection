import os
import gdown
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from PIL import Image

# -----------------------------
# CONFIG: Google Drive model
# -----------------------------
# direct-download link for your uploaded best_saved.pt
MODEL_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "best_saved.pt"

# -----------------------------
# Helper: download model if missing
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (this may take a minute)..."):
            gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded.")

# -----------------------------
# Load YOLO model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    download_model()
    # Return ultralytics YOLO wrapper
    return YOLO(MODEL_PATH)

# Load once
model = load_model()

# -----------------------------
# Label -> color mapping (BGR for OpenCV)
# -----------------------------
CATEGORY_COLOR = {
    "organic": (0, 255, 0),      # green
    "anorganic": (0, 255, 255),  # yellow (BGR)
    "b3": (0, 0, 255)            # red
}

def get_color(label: str):
    return CATEGORY_COLOR.get(label.lower(), (255, 255, 255))

# -----------------------------
# Draw boxes for a frame (numpy BGR)
# -----------------------------
def draw_boxes_frame(img: np.ndarray, results) -> np.ndarray:
    """
    img: BGR numpy array
    results: ultralytics result object (call model(img))
    returns annotated BGR numpy array
    """
    out = img.copy()
    # results can contain multiple result objects; usually results[0]
    for r in results:
        # r.boxes is a Boxes object
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            # xyxy, cls, conf
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            # convert name
            name = r.names[cls_idx] if (hasattr(r, "names") and r.names is not None and cls_idx in r.names) else str(cls_idx)
            name_low = name.lower()
            color = get_color(name_low)
            # draw rectangle and text
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
            label = f"{name.upper()} {conf:.2f}"
            # text background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Waste Detection - Realtime", layout="wide")
st.title("♻️ Waste Detection — Realtime & Upload")
st.write("Realtime detection: pegang sampah ke kamera, kotak deteksi akan tampil langsung. Upload juga didukung.")

tab1, tab2 = st.tabs(["🎥 Realtime (Camera)", "🖼 Upload Image"])

# -----------------------------
# WEbrtc realtime processor
# -----------------------------
class RTProcessor(VideoProcessorBase):
    def __init__(self):
        # store optional state
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # receive BGR frame
        img = frame.to_ndarray(format="bgr24")
        # run model (inference)
        # Note: calling model(img) returns a Results object or list of Results
        results = model(img, verbose=False)  # keep verbose False to minimize logs
        # draw boxes
        annotated = draw_boxes_frame(img, results)
        # convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# choose camera facing mode option for user (environment = back, user = front)
with tab1:
    st.subheader("Realtime Camera")
    col1, col2 = st.columns([1, 3])
    with col1:
        facing = st.selectbox("Pilih Kamera (facingMode)", ["environment (Belakang)", "user (Depan)"])
        btn = st.button("Start Realtime")
    with col2:
        st.caption("Arahkan sampah ke kamera, deteksi akan muncul live di layar.")

    if btn:
        facing_mode = "environment" if facing.startswith("environment") else "user"

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="waste-detector-webrtc",
            mode="recvonly",
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "facingMode": facing_mode
                },
                "audio": False
            },
            video_processor_factory=RTProcessor,
            async_processing=True,
        )

# -----------------------------
# UPLOAD IMAGE TAB
# -----------------------------
with tab2:
    st.subheader("Upload foto atau ambil menggunakan kamera (open camera pada HP)")
    uploaded = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Original Image", use_column_width=True)

        # convert PIL -> numpy BGR for model (ultralytics accepts PIL/numpy too)
        img_np = np.array(img_pil)[:, :, ::-1]  # RGB->BGR
        results = model(img_np)
        annotated = draw_boxes_frame(img_np, results)

        # convert BGR->RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

# -----------------------------
# Footer / tips
# -----------------------------
st.markdown("---")
st.write("Tips: gunakan kamera belakang untuk hasil lebih tajam. Jika deteksi lambat, turunkan resolusi pada media_stream_constraints.")
