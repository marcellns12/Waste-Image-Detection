import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Fixed", layout="centered")

# CSS agar tampilan lebih rapi
st.markdown("""
    <style>
    .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. KONFIGURASI API (Ganti API Key jika perlu)
# ==================================================
API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# Inisialisasi Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ==================================================
# 3. SETTING SENSITIVITAS (PENTING!)
# ==================================================
st.sidebar.header("⚙️ Pengaturan Deteksi")
st.sidebar.info("Jika hasil deteksi salah (selalu B3), naikkan nilai ini.")

# Slider untuk mengatur 'Confidence Threshold'
# Default 0.4 (40%). Objek dengan keyakinan di bawah ini akan diabaikan.
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold (Keyakinan)", 0.0, 1.0, 0.40, 0.05)

# ==================================================
# 4. FUNGSI LOGIKA UTAMA
# ==================================================
def draw_predictions(image, predictions, threshold):
    """Menggambar kotak HANYA jika confidence > threshold"""
    if not predictions or 'predictions' not in predictions:
        return image, 0

    count = 0
    for box in predictions['predictions']:
        conf = box['confidence']
        
        # --- FILTER PENTING ---
        # Jika keyakinan AI di bawah settingan slider, lewati (jangan gambar)
        if conf < threshold:
            continue
            
        count += 1
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        label = box['class']

        # Koordinat
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Warna Kotak (Hijau Stabilo)
        color = (0, 255, 0)

        # Gambar Kotak
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label Background
        text = f"{label} {conf:.0%}"
        (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 25), (x1 + t_w, y1), color, -1)
        
        # Tulis Text
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image, count

def process_image(image_bytes, source_type="upload"):
    """Fungsi tunggal untuk memproses gambar dari Upload maupun Kamera"""
    temp_path = None
    try:
        # 1. Baca Gambar dari Bytes
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1) # Load as BGR

        # Validasi jika gambar rusak
        if img_bgr is None:
            st.error("Gagal membaca file gambar.")
            return

        # 2. Simpan Sementara (Wajib buat API Roboflow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img_bgr)

        # 3. Kirim ke API (Tampilkan Loading)
        with st.spinner(f'Sedang menganalisis {source_type}...'):
            result = CLIENT.infer(temp_path, model_id=MODEL_ID)

        # 4. Gambar Kotak (Sesuai Slider Threshold)
        annotated_img, count = draw_predictions(img_bgr, result, CONFIDENCE_THRESHOLD)

        # 5. Tampilkan Hasil
        st.success(f"Selesai! Terdeteksi: {count} objek (Diatas {CONFIDENCE_THRESHOLD*100:.0f}%)")
        
        # Convert BGR ke RGB untuk Streamlit
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Debugging JSON (Opsional, taruh di expander biar rapi)
        with st.expander("Lihat Data Mentah (JSON)"):
            st.json(result)

    except Exception as e:
        st.error(f"Terjadi Kesalahan: {e}")
    finally:
        # Bersihkan file sampah
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ==================================================
# 5. UI UTAMA (TABS)
# ==================================================
st.title("♻️ Waste Detection App")
st.markdown("Aplikasi deteksi sampah menggunakan Roboflow API.")

tab1, tab2 = st.tabs(["📸 Ambil Foto (Kamera)", "🖼️ Upload File"])

# --- TAB 1: KAMERA (Diperbaiki) ---
with tab1:
    st.write("Klik tombol di bawah untuk mengambil gambar.")
    
    # Kunci: Gunakan key unik agar tidak bentrok
    camera_file = st.camera_input("Buka Kamera", key="camera_input")
    
    if camera_file is not None:
        # Langsung proses
        process_image(camera_file.getvalue(), source_type="Kamera")

# --- TAB 2: UPLOAD ---
with tab2:
    uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if st.button("🔍 Mulai Deteksi", key="btn_upload"):
            process_image(uploaded_file.getvalue(), source_type="Upload")
