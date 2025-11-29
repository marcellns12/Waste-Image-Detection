import streamlit as st
import cv2
import numpy as np
import gdown
import os
from ultralytics import YOLO

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection App", layout="centered")

# ID File Google Drive & Nama File
DRIVE_FILE_ID = '1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid'
MODEL_FILENAME = 'my_waste_model.pt'

# ==================================================
# 2. LOAD MODEL (Otomatis Download)
# ==================================================
@st.cache_resource
def load_model():
    # Cek apakah file sudah ada, jika belum download dari Drive
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Sedang mendownload model dari Google Drive..."):
            try:
                url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
                gdown.download(url, MODEL_FILENAME, quiet=False)
            except Exception as e:
                st.error(f"Gagal mendownload model: {e}")
                return None

    # Load Model YOLO
    try:
        model = YOLO(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"File model rusak atau tidak kompatibel: {e}")
        return None

model = load_model()

# ==================================================
# 3. SIDEBAR (Hanya Slider Threshold)
# ==================================================
st.sidebar.header("⚙️ Pengaturan")

# Slider untuk memfilter hasil yang kurang yakin
# Default 0.50 (50%). Jika di bawah ini, objek tidak akan ditampilkan.
CONF_THRESHOLD = st.sidebar.slider("Akurasi Minimum (Threshold)", 0.0, 1.0, 0.50, 0.05)

# ==================================================
# 4. FUNGSI DETEKSI (PROCESS IMAGE)
# ==================================================
def process_image(image_bytes):
    if model is None: return None, []

    try:
        # 1. Baca data gambar (Decode)
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
            return None, []

        # 2. Convert BGR ke RGB (PENTING: Agar warna tidak tertukar)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 3. Prediksi menggunakan YOLO
        results = model.predict(img_rgb, conf=CONF_THRESHOLD)
        
        # 4. Gambar Kotak (Plotting)
        # results[0].plot() mengembalikan format BGR
        annotated_bgr = results[0].plot()
        
        # 5. Convert hasil plot ke RGB untuk ditampilkan
        final_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Ambil nama kelas yang terdeteksi untuk laporan teks
        detected_classes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            # Format teks: "B3 (85%)"
            detected_classes.append(f"{class_name} ({conf:.0%})")
        
        return final_img, detected_classes

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, []

# ==================================================
# 5. TAMPILAN UTAMA (UI)
# ==================================================
st.title("♻️ Waste Detection App")
st.markdown("Aplikasi deteksi sampah (Model Lokal).")

if model is None:
    st.warning("⚠️ Model gagal dimuat. Coba refresh halaman.")
    st.stop()

# Membuat Tab
tab1, tab2 = st.tabs(["📸 Kamera", "🖼️ Upload File"])

# --- TAB 1: KAMERA ---
with tab1:
    st.info("Pastikan cahaya cukup agar deteksi akurat.")
    camera_file = st.camera_input("Ambil Foto", key="cam_input")
    
    if camera_file:
        final_img, classes = process_image(camera_file.getvalue())
        
        if final_img is not None:
            if not classes:
                st.warning("⚠️ Objek tidak terdeteksi. Coba dekatkan objek atau turunkan Threshold di sidebar.")
                st.image(final_img, caption="Gambar Asli", use_column_width=True)
            else:
                # Tampilkan hasil unik (set) agar tidak spam nama yang sama
                unique_classes = list(set([x.split(' (')[0] for x in classes])) 
                st.success(f"Terdeteksi: {', '.join(unique_classes)}")
                st.image(final_img, caption="Hasil Deteksi", use_column_width=True)

# --- TAB 2: UPLOAD ---
with tab2:
    uploaded = st.file_uploader("Upload foto (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded:
        if st.button("🔍 Mulai Deteksi"):
            final_img, classes = process_image(uploaded.read())
            
            if final_img is not None:
                if not classes:
                    st.warning("⚠️ Objek tidak terdeteksi.")
                    st.image(final_img, caption="Gambar Asli", use_column_width=True)
                else:
                    unique_classes = list(set([x.split(' (')[0] for x in classes]))
                    st.success(f"Terdeteksi: {', '.join(unique_classes)}")
                    st.image(final_img, caption="Hasil Deteksi", use_column_width=True)
