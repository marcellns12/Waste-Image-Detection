import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="♻️ Waste Detection Final", layout="centered")

# API Configuration
API_KEY = "ItgMPolGq0yMOI4nhLpe"
MODEL_ID = "waste-project-pgzut/3"

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ==================================================
# 2. SETUP WARNA (CUSTOM COLOR)
# ==================================================
# ANORGANIK = KUNING
annotator_anorganik_box = sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=4)
annotator_anorganik_label = sv.LabelAnnotator(color=sv.Color.YELLOW, text_scale=0.8, text_thickness=2, text_color=sv.Color.BLACK)

# ORGANIK = HIJAU
annotator_organik_box = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=4)
annotator_organik_label = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.8, text_thickness=2, text_color=sv.Color.BLACK)

# B3 = MERAH
annotator_b3_box = sv.BoxAnnotator(color=sv.Color.RED, thickness=4)
annotator_b3_label = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.8, text_thickness=2, text_color=sv.Color.WHITE)

# ==================================================
# 3. FUNGSI LOGIKA DETEKSI
# ==================================================
def process_image(image_buffer, threshold):
    try:
        # 1. Baca gambar dari buffer kamera
        file_bytes = np.asarray(bytearray(image_buffer.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
             return None, 0, {}

        # 2. Kirim ke API Roboflow
        result = CLIENT.infer(img_bgr, model_id=MODEL_ID)

        # 3. Konversi ke Format Supervision
        detections = sv.Detections.from_inference(result)

        # ---------------------------------------------------------
        # [FIX PENTING] FILTERING CONFIDENCE
        # ---------------------------------------------------------
        # Cek kosong dulu
        if len(detections) == 0:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, result

        # Gunakan np.array() untuk memastikan tipe datanya benar saat dibandingkan
        # Kita pakai nilai 'threshold' dari Slider
        valid_idx = np.array(detections.confidence) > threshold
        detections = detections[valid_idx]
        
        # Cek lagi setelah difilter
        if len(detections) == 0:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0, result
        # ---------------------------------------------------------

        # 4. PEWARNAAN BERDASARKAN KELAS
        annotated_image = img_bgr.copy()

        # --- A. Warnai ANORGANIK (Kuning) ---
        # Logic: Namanya mengandung "anorganik"
        mask_anorganik = np.array([
            "anorganik" in data["class_name"].lower() 
            for data in detections.data.values()
        ])
        if mask_anorganik.any():
            det_temp = detections[mask_anorganik]
            annotated_image = annotator_anorganik_box.annotate(scene=annotated_image, detections=det_temp)
            annotated_image = annotator_anorganik_label.annotate(scene=annotated_image, detections=det_temp)

        # --- B. Warnai ORGANIK (Hijau) ---
        # Logic: Namanya mengandung "organik" TAPI BUKAN "anorganik"
        mask_organik = np.array([
            "organik" in data["class_name"].lower() and "anorganik" not in data["class_name"].lower()
            for data in detections.data.values()
        ])
        if mask_organik.any():
            det_temp = detections[mask_organik]
            annotated_image = annotator_organik_box.annotate(scene=annotated_image, detections=det_temp)
            annotated_image = annotator_organik_label.annotate(scene=annotated_image, detections=det_temp)

        # --- C. Warnai B3 (Merah) ---
        # Logic: Namanya mengandung "b3"
        mask_b3 = np.array([
            "b3" in data["class_name"].lower() 
            for data in detections.data.values()
        ])
        if mask_b3.any():
            det_temp = detections[mask_b3]
            annotated_image = annotator_b3_box.annotate(scene=annotated_image, detections=det_temp)
            annotated_image = annotator_b3_label.annotate(scene=annotated_image, detections=det_temp)

        # 5. Return Hasil (Convert ke RGB buat Streamlit)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), len(detections), result

    except Exception as e:
        st.error(f"System Error: {e}")
        return None, 0, {}

# ==================================================
# 4. TAMPILAN UTAMA (UI)
# ==================================================
st.title("♻️ Smart Waste Detection")
st.markdown("🔴 **B3** | 🟡 **Anorganik** | 🟢 **Organik**")

# --- SLIDER SENSITIVITAS (SOLUSI BOTOL TIDAK TERDETEKSI) ---
st.info("💡 **Tips:** Jika botol transparan tidak terdeteksi, **geser slider ke kiri (0.15 - 0.20)**.")
conf_threshold = st.slider("Tingkat Keyakinan (Confidence Threshold)", 0.05, 1.0, 0.25, 0.05)

# --- KAMERA INPUT ---
camera_file = st.camera_input("Ambil Foto", label_visibility="collapsed")

if camera_file is not None:
    # Tampilkan Spinner loading
    with st.spinner("Sedang menganalisis gambar..."):
        final_img, count, raw_json = process_image(camera_file, conf_threshold)
    
    if final_img is not None:
        st.image(final_img, caption="Hasil Deteksi", use_column_width=True)
        
        if count > 0:
            st.success(f"✅ Berhasil mendeteksi {count} jenis sampah!")
        else:
            st.warning("⚠️ Objek tidak terdeteksi.")
            st.write(f"Sistem mencoba mencari dengan keyakinan di atas **{conf_threshold*100}%**, tapi tidak menemukannya.")
            st.write("👉 Coba **turunkan slider** di atas atau perbaiki pencahayaan.")
            
            # Debugging Tools
            with st.expander("Lihat Data Mentah AI (Untuk Pengecekan)"):
                st.json(raw_json)
