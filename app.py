import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI API & UI
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Stable", layout="centered")

API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3" # Model ID Anda

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# --------------------------------------------------
# Setup Annotator (Warna Custom)
# --------------------------------------------------
# A. ANORGANIK = KUNING (Yellow)
annotator_anorganik_box = sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=4)
annotator_anorganik_label = sv.LabelAnnotator(color=sv.Color.YELLOW, text_scale=0.8, text_thickness=2, text_color=sv.Color.BLACK)

# B. ORGANIK = HIJAU (Green)
annotator_organik_box = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=4)
annotator_organik_label = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.8, text_thickness=2, text_color=sv.Color.BLACK)

# C. B3 = MERAH (Red)
annotator_b3_box = sv.BoxAnnotator(color=sv.Color.RED, thickness=4)
annotator_b3_label = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2, text_color=sv.Color.WHITE)


# ==================================================
# 3. FUNGSI LOGIKA DETEKSI (ANTI-CRASH)
# ==================================================
def process_image(image_buffer, threshold):
    try:
        # 1. Baca gambar dari buffer kamera
        file_bytes = np.asarray(bytearray(image_buffer.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
             return None, 0, {}

        # 2. Kirim ke API Roboflow
        with st.spinner("Menganalisis jenis sampah..."):
            result = CLIENT.infer(img_bgr, model_id=MODEL_ID)

        # 3. Konversi ke Supervision
        detections = sv.Detections.from_inference(result)

        # 4. FILTERING CONFIDENCE (FIXED & ROBUST)
        if len(detections) == 0:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, result
        
        # Lakukan filtering dengan numpy array (FIX ERROR INDEX)
        valid_idx = np.array(detections.confidence) > threshold
        detections = detections[valid_idx]
        
        if len(detections) == 0:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, result

        # 5. PEWARNAAN BERDASARKAN KELAS (Menggunakan Filtering Supervision)
        annotated_image = img_bgr.copy()
        total_detected = 0

        # --- A. ANORGANIK (Kuning) ---
        # Diasumsikan nama kelas di model adalah "Anorganik" (atau sejenisnya)
        # Kita akan gunakan filter_by_class_name
        det_anorganik = detections.filter_by_class_name(classes=["Anorganik", "anorganik", "Botol", "Plastik"])
        if len(det_anorganik) > 0:
            annotated_image = annotator_anorganik_box.annotate(scene=annotated_image, detections=det_anorganik)
            annotated_image = annotator_anorganik_label.annotate(scene=annotated_image, detections=det_anorganik)
            total_detected += len(det_anorganik)

        # --- B. ORGANIK (Hijau) ---
        det_organik = detections.filter_by_class_name(classes=["Organik", "organik", "Daun", "Makanan"])
        if len(det_organik) > 0:
            annotated_image = annotator_organik_box.annotate(scene=annotated_image, detections=det_organik)
            annotated_image = annotator_organik_label.annotate(scene=annotated_image, detections=det_organik)
            total_detected += len(det_organik)

        # --- C. B3 (Merah) ---
        det_b3 = detections.filter_by_class_name(classes=["B3", "b3", "Baterai", "Kimia"])
        if len(det_b3) > 0:
            annotated_image = annotator_b3_box.annotate(scene=annotated_image, detections=det_b3)
            annotated_image = annotator_b3_label.annotate(scene=annotated_image, detections=det_b3)
            total_detected += len(det_b3)

        # 6. Return Hasil
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), total_detected, result

    except Exception as e:
        st.error(f"System Error: {e}")
        # Jika error, tampilkan gambar asli
        if 'img_bgr' in locals() and img_bgr is not None:
             return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, {}
        return None, 0, {}

# ==================================================
# 4. TAMPILAN UTAMA (UI)
# ==================================================
st.title("♻️ Smart Waste Detection")
st.write("🔴 **B3** | 🟡 **Anorganik** | 🟢 **Organik**")

# SLIDER SENSITIVITAS (Dibutuhkan untuk mendeteksi botol transparan)
st.info("💡 **Tips:** Untuk botol transparan yang sulit dideteksi, **geser Threshold ke 0.15 - 0.25**.")
conf_threshold = st.slider("Tingkat Keyakinan (Confidence Threshold)", 0.05, 1.0, 0.25, 0.05)

# Kamera Input
camera_file = st.camera_input("Ambil Foto", label_visibility="collapsed")

if camera_file is not None:
    # Proses gambar
    with st.spinner("Sedang menganalisis gambar..."):
        final_img, count, raw_json = process_image(camera_file, conf_threshold)
    
    if final_img is not None:
        st.image(final_img, caption="Hasil Deteksi", use_column_width=True)
        
        if count > 0:
            st.success(f"✅ Berhasil mendeteksi {count} objek!")
        else:
            st.warning("⚠️ Objek tidak terdeteksi.")
            # Debugging tools
            with st.expander("Lihat Data Mentah AI"):
                st.json(raw_json)
