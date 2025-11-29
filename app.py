import streamlit as st
import cv2
import tempfile
import gdown
import os
from ultralytics import YOLO

st.set_page_config(page_title="Waste Detection", layout="wide")

# ----------------------------------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# ----------------------------------------------------
DRIVE_URL = "https://drive.google.com/file/d/1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid/view?usp=sharing"
MODEL_PATH = "waste_model.pt"


@st.cache_resource
def download_and_load_model():
    # Download model from Google Drive if missing
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive… please wait…")
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    return model


model = download_and_load_model()


# ----------------------------------------------------
# COLOR MAPPING
# ----------------------------------------------------
COLORS = {
    "plastic": (0, 255, 0),
    "paper": (255, 0, 0),
    "metal": (0, 0, 255),
    "glass": (255, 255, 0),
    "organic": (0, 255, 255),
    "other": (255, 0, 255),
}


def draw_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = r.names[cls]
            color = COLORS.get(name, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return frame


# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.title("♻️ Waste Detection with YOLO")
st.write("Letakkan sampah di depan kamera — model akan mendeteksi secara realtime.")


# ----------------------------------------------------
# Live Webcam
# ----------------------------------------------------
run_webcam = st.checkbox("Aktifkan Webcam")

if run_webcam:
    stframe = st.empty()

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        st.error("Gagal membuka webcam! Berikan izin kamera terlebih dahulu.")
    else:
        st.success("Webcam aktif! Arahkan sampah ke kamera 👇")

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak bisa membaca frame dari kamera.")
            break

        # YOLO inference
        results = model(frame)

        # Draw bounding boxes
        frame = draw_boxes(frame, results)

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display
        stframe.image(frame, channels="RGB")

    cap.release()
