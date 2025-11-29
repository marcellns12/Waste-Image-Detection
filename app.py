import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Fixed", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ==================================================
# 2. SETUP WARNA (CUSTOM COLOR)
# ==================================================
# A. ANORGANIK = KUNING
annotator_anorganik_box = sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=4)
annotator_anorganik_label = sv.LabelAnnotator(color=sv.Color.YELLOW, text_scale=0.8, text_thickness=2)

# B. ORGANIK = HIJAU
annotator_organik_box = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=4)
annotator_organik_label = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.8, text_thickness=2)

# C. B3 = MERAH
annotator_b3_box = sv.BoxAnnotator(color=sv.Color.RED, thickness=4)
annotator_b3_label = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2)


# ==================================================
# 3. FUNGSI LOGIKA DETEKSI (FIXED)
# ==================================================
def process_image(image_buffer):
    try:
        # 1. Baca gambar
        file_bytes = np.asarray(bytearray(image_buffer.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
             st.error("Gambar tidak terbaca.")
             return None, 0

        # 2. Kirim ke API Roboflow
        # [PERBAIKAN] Hapus parameter 'confidence' di sini karena bikin error
        with st.spinner("Menganalisis jenis sampah..."):
            result = CLIENT.infer(img_bgr, model_id=MODEL_ID)

        # 3. Konversi ke Supervision
        detections = sv.Detections.from_inference(result)

        # [PERBAIKAN] Filter Confidence dilakukan DI SINI
        # Hanya ambil yang keyakinannya > 30% (0.3)
        detections = detections[detections.confidence > 0.3]

        # 4. FILTERING & PEWARNAAN
        # Kita copy gambar asli biar tidak rusak saat ditimpa warna
        annotated_image = img_bgr.copy()

        # --- A. Warnai ANORGANIK (Kuning) ---
        mask_anorganik = np.array([
            "anorganik" in data["class_name"].lower() 
            for data in detections.data.values()
        ])
        if mask_anorganik.any():
            det_anorganik = detections[mask_anorganik]
            annotated_image = annotator_anorganik_box.annotate(scene=annotated_image, detections=det_anorganik)
            annotated_image = annotator_anorganik_label.annotate(scene=annotated_image, detections=det_anorganik)

        # --- B. Warnai ORGANIK (Hijau) ---
        mask_organik = np.array([
            "organik" in data["class_name"].lower() and "anorganik" not in data["class_name"].lower()
            for data in detections.data.values()
        ])
        if mask_organik.any():
            det_organik = detections[mask_organik]
            annotated_image = annotator_organik_box.annotate(scene=annotated_image, detections=det_organik)
            annotated_image = annotator_organik_label.annotate(scene=annotated_image, detections=det_organik)

        # --- C. Warnai B3 (Merah) ---
        mask_b3 = np.array([
            "b3" in data["class_name"].lower() 
            for data in detections.data.values()
        ])
        if mask_b3.any():
            det_b3 = detections[mask_b3]
            annotated_image = annotator_b3_box.annotate(scene=annotated_image, detections=det_b3)
            annotated_image = annotator_b3_label.annotate(scene=annotated_image, detections=det_b3)

        # 5. Return Hasil (Convert BGR -> RGB)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), len(detections)

    except Exception as e:
        st.error(f"Error: {e}")
        return None, 0

# ==================================================
# 4. TAMPILAN UTAMA
# ==================================================
st.title("♻️ Smart Waste Detection")
st.write("🔴 **B3** | 🟡 **Anorganik** | 🟢 **Organik**")

# Kamera Input
camera_file = st.camera_input("Ambil Foto", key="cam_final", label_visibility="collapsed")

if camera_file is not None:
    final_img, count = process_image(camera_file)
    
    if final_img is not None:
        st.image(final_img, caption="Hasil Deteksi Berwarna", use_column_width=True)
        
        if count > 0:
            st.success(f"✅ Selesai! Terdeteksi {count} objek.")
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi.")
