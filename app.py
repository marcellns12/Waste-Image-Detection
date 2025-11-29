import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from PIL import Image

# ==================================================
# 1. KONFIGURASI
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Stabil", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# Inisialisasi Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Setup Annotator (Supaya kotak hasilnya rapi & bagus)
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_padding=10)

# ==================================================
# 2. FUNGSI LOGIKA (DENGAN RESIZE)
# ==================================================
def process_image(image_buffer):
    try:
        # 1. Baca gambar dari buffer kamera
        file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        # 2. [PENTING] Resize Gambar biar Internet Gak Berat
        # Kita kecilkan ke lebar 640px. Kualitas tetap bagus, tapi size file turun 80%.
        # Ini kunci agar tidak "Error Connection" ke API.
        height, width = img_bgr.shape[:2]
        new_width = 640
        new_height = int((new_width / width) * height)
        img_resized = cv2.resize(img_bgr, (new_width, new_height))

        # 3. Kirim ke Roboflow API
        with st.spinner("Mengidentifikasi jenis sampah..."):
            result = CLIENT.infer(img_resized, model_id=MODEL_ID)

        # 4. Konversi Hasil ke Format Supervision
        detections = sv.Detections.from_inference(result)

        # 5. Filter Sampah yang "Tidak Yakin" (Threshold)
        # Hilangkan deteksi yang keyakinannya di bawah 40% (0.4) biar gak asal tebak B3
        detections = detections[detections.confidence > 0.4]

        # 6. Gambar Kotak & Label
        annotated_image = box_annotator.annotate(scene=img_resized.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # 7. Return Hasil (Convert ke RGB buat Streamlit)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), len(detections)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None, 0

# ==================================================
# 3. TAMPILAN UTAMA
# ==================================================
st.title("♻️ Waste Detection (Mode Stabil)")
st.markdown("Mode ini lebih cepat dan **anti-error** koneksi.")

col1, col2 = st.columns(2)

# --- KOLOM 1: KAMERA ---
with col1:
    st.write("📷 **Ambil Foto**")
    # Camera Input (Native Streamlit) - Sangat Stabil
    camera_file = st.camera_input("Klik tombol untuk mendeteksi", label_visibility="collapsed")

# --- KOLOM 2: HASIL ---
with col2:
    st.write("🤖 **Hasil Deteksi**")
    
    if camera_file is not None:
        # Proses gambar segera setelah dijepret
        final_img, count = process_image(camera_file)
        
        if final_img is not None:
            st.image(final_img, use_column_width=True)
            
            if count > 0:
                st.success(f"Berhasil mendeteksi {count} objek!")
            else:
                st.warning("Tidak ada objek terdeteksi. Coba dekatkan kamera.")
    else:
        # Placeholder jika belum ada foto
        st.info("Hasil akan muncul di sini setelah Anda mengambil foto.")
