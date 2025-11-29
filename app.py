import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import io

st.set_page_config(page_title="Waste Detection Mobile", layout="wide")


# ===========================================
# LOAD MODEL
# ===========================================
@st.cache_resource
def load_model():
    model_path = "yolo11_best_weights.pt"  # Ganti sesuai filenya
    return YOLO(model_path)

model = load_model()


# ===========================================
# KATEGORI + WARNA
# ===========================================
CATEGORY_COLOR = {
    "organic": (0, 255, 0),       # hijau
    "anorganic": (0, 255, 255),   # kuning
    "b3": (0, 0, 255)             # merah
}

def get_color(lbl):
    return CATEGORY_COLOR.get(lbl.lower(), (255, 255, 255))


# ===========================================
# DRAW BOX
# ===========================================
def draw_boxes(image, results):
    img = np.array(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            cls_name = r.names[cls].lower()
            conf = float(box.conf[0])

            color = get_color(cls_name)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(img, f"{cls_name.upper()} ({conf:.2f})",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img


# ===========================================
# UI
# ===========================================
st.title("📱 Waste Detection – Realtime Capture & Upload")
st.write("Deteksi sampah **Organik (Hijau)**, **Anorganik (Kuning)**, dan **B3 (Merah)** dari kamera HP atau upload foto.")


tab1, tab2 = st.tabs(["📸 Kamera HP", "🖼️ Upload Foto"])


# ===========================================
# TAB 1 – CAMERA CAPTURE
# ===========================================
with tab1:
    st.subheader("Ambil Foto dari Kamera HP")

    st.write("Klik tombol kamera di bawah untuk mengambil gambar → otomatis muncul hasilnya.")

    image_data = st.file_uploader("Foto dari Kamera HP", 
                                  type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=False,
                                  key="camera_input",
                                  label_visibility="collapsed")

    # HTML hack agar membuka kamera HP
    st.markdown("""
    <input type="file" accept="image/*" capture="environment" 
           onchange="document.querySelector('input[type=file]').dispatchEvent(new Event('input'));">
    """, unsafe_allow_html=True)

    if image_data is not None:
        image = Image.open(image_data).convert("RGB")

        st.image(image, caption="Foto Kamera", use_column_width=True)

        result = model(image)
        annotated = draw_boxes(image, result)

        st.image(annotated, caption="Hasil Deteksi Sampah", use_column_width=True)


# ===========================================
# TAB 2 – UPLOAD FOTO MANUAL
# ===========================================
with tab2:
    st.subheader("Upload Foto dari Galeri")

    img_file = st.file_uploader("Upload Foto:", type=["jpg", "jpeg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")

        st.image(image, caption="Gambar Asli", use_column_width=True)

        result = model(image)
        annotated = draw_boxes(image, result)

        st.image(annotated, caption="Hasil Deteksi Sampah", use_column_width=True)
