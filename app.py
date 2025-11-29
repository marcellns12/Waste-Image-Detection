import streamlit as st
import cv2
import numpy as np
import gdown
import os
from ultralytics import YOLO

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection (Drive Model)", layout="centered")

# ID File Google Drive Kamu
# Link asli: https://drive.google.com/file/d/1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid/view
DRIVE_FILE_ID = '1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid'
MODEL_FILENAME = 'waste_model.pt'

# ==================================================
# 2. FUNGSI DOWNLOAD & LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    """
    Mendownload model dari Google Drive jika belum ada, lalu me-load ke memori.
    Menggunakan cache agar tidak download ulang setiap refresh.
    """
    # 1. Cek apakah file model sudah ada
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner(f"Sedang mendownload model dari Google Drive... (Harap tunggu)"):
            try:
                # URL download gdown
                url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
                gdown.download(url, MODEL_FILENAME, quiet=False)
                st.success("Download model berhasil!")
            except Exception as e:
                st.error(f"Gagal mendownload model: {e}")
                return None

    # 2. Load Model YOLO
    try:
        model = YOLO(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

# Load model di awal
model = load_model()

# ==================================================
# 3. SETTING SENSITIVITAS
# ==================================================
st.sidebar.header("⚙️ Pengaturan")
# Slider Confidence
CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold (Keyakinan)", 0.0, 1.0, 0.40, 0.05)

# ==================================================
# 4. FUNGSI DETEKSI (LOKAL)
# ==================================================
def process_image(image_bytes):
    """
    Menerima bytes gambar -> Prediksi Lokal (YOLO) -> Return Gambar Hasil
    """
    if model is None:
        st.error("Model belum dimuat.")
        return None, None

    try:
        # 1. Decode gambar ke OpenCV
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
            return None, None

        # 2. Prediksi menggunakan Model Lokal
        # conf=... mengatur batas keyakinan agar tidak asal tebak
        results = model.predict(img_bgr, conf=CONF_THRESHOLD)
        
        # 3. Ambil gambar hasil plotting (sudah ada kotaknya)
        # results[0].plot() mengembalikan array BGR
        annotated_img = results[0].plot()

        # 4. Convert BGR ke RGB untuk Streamlit
        final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Hitung jumlah objek
        count = len(results[0].boxes)
        
        # Ambil nama class yang terdeteksi untuk info
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        
        return final_img, detected_classes

    except Exception as e:
        st.error(f"Error proses deteksi: {e}")
        return None, None

# ==================================================
# 5. UI UTAMA
# ==================================================
st.title("♻️ Waste Detection (Local Model)")
st.markdown("Model didownload langsung dari Google Drive dan dijalankan di server ini.")

if model is None:
    st.warning("⚠️ Model gagal dimuat. Pastikan File ID Google Drive benar dan bisa diakses publik.")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["📸 Kamera", "🖼️ Upload File"])

# --- TAB 1: KAMERA ---
with tab1:
    st.info("Ambil foto untuk mendeteksi.")
    camera_file = st.camera_input("Kamera", key="cam")
    
    if camera_file:
        final_img, classes = process_image(camera_file.getvalue())
        if final_img is not None:
            st.success(f"Terdeteksi: {', '.join(set(classes))} ({len(classes)} objek)")
            st.image(final_img, use_column_width=True)

# --- TAB 2: UPLOAD ---
with tab2:
    uploaded_file = st.file_uploader("Upload Foto", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        if st.button("🔍 Deteksi"):
            final_img, classes = process_image(uploaded_file.read())
            if final_img is not None:
                st.success(f"Terdeteksi: {', '.join(set(classes))} ({len(classes)} objek)")
                st.image(final_img, use_column_width=True)
