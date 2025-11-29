import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI HALAMAN & API
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection API", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

# Inisialisasi Client Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ==================================================
# 2. FUNGSI GAMBAR KOTAK (VISUALISASI)
# ==================================================
def draw_predictions(image, predictions):
    """
    Menggambar bounding box di atas gambar berdasarkan respon JSON Roboflow.
    """
    # Cek apakah ada prediksi
    if 'predictions' not in predictions:
        return image

    for box in predictions['predictions']:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        label = box['class']
        conf = box['confidence']

        # Hitung koordinat
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Warna (Hijau)
        color = (0, 255, 0) 

        # 1. Gambar Kotak
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 2. Gambar Label Background
        text = f"{label} ({conf:.1%})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 25), (x1 + text_w, y1), color, -1)

        # 3. Tulis Teks
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image

# ==================================================
# 3. FUNGSI PROSES UTAMA
# ==================================================
def process_image(image_bytes):
    """
    Menerima bytes gambar -> Simpan Temp -> Kirim API -> Gambar Kotak -> Return RGB
    """
    # 1. Decode bytes ke OpenCV Image (BGR)
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    # 2. Simpan ke file temporary (Wajib karena SDK butuh path file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, img_bgr)

    try:
        # 3. Kirim ke Roboflow API
        with st.spinner('Sedang mengirim ke server Roboflow...'):
            result = CLIENT.infer(temp_path, model_id=MODEL_ID)
        
        # 4. Gambar hasil ke foto
        annotated_bgr = draw_predictions(img_bgr, result)

        # 5. Convert ke RGB untuk Streamlit
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), result

    except Exception as e:
        st.error(f"Error API: {e}")
        return None, None
    finally:
        # Hapus file sampah
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==================================================
# 4. UI STREAMLIT
# ==================================================
st.title("♻️ Waste Detection (Roboflow Cloud)")
st.markdown(f"Menggunakan Model ID: `{MODEL_ID}`")

# Tab Pilihan
tab1, tab2 = st.tabs(["🖼️ Upload Foto", "📷 Ambil Foto (Webcam)"])

# --- TAB 1: UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Upload foto sampah (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        if st.button("🔍 Deteksi File Ini"):
            final_img, json_res = process_image(uploaded_file.read())
            if final_img is not None:
                st.image(final_img, caption="Hasil Deteksi", use_column_width=True)
                # Opsional: Tampilkan data mentah
                with st.expander("Lihat Data JSON"):
                    st.json(json_res)

# --- TAB 2: WEBCAM SNAPSHOT ---
with tab2:
    st.info("Klik tombol 'Take Photo' di bawah untuk mengambil gambar.")
    
    # st.camera_input jauh lebih stabil daripada webrtc_streamer untuk kasus API
    camera_file = st.camera_input("Kamera")
    
    if camera_file is not None:
        # Langsung proses otomatis setelah foto diambil
        final_img, json_res = process_image(camera_file.
