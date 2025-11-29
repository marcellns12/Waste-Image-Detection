import cv2
import torch
import numpy as np
import urllib.request
import os

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1DSp9TnRA_twScvL-Ds84vrMO9iYOPfid"
MODEL_PATH = "waste_model.pt"

# ----------------------------
# Download model if not exists
# ----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⏳ Downloading model from Google Drive...")
        urllib.request.urlretrieve(MODEL_DRIVE_URL, MODEL_PATH)
        print("✅ Model downloaded!")

# ----------------------------
# Load YOLO model
# ----------------------------
def load_model():
    print("🔄 Loading model...")
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    print("✅ Model loaded successfully!")
    return model

# ----------------------------
# Draw boxes
# ----------------------------
def draw_boxes(frame, results):
    for *box, conf, cls in results:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# ----------------------------
# Main detection loop
# ----------------------------
if __name__ == "__main__":

    print("♻️ Waste Detection with YOLO")
    print("Letakkan sampah di depan kamera — deteksi realtime aktif.")
    print()

    download_model()
    model = load_model()

    # OPEN CAMERA
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("❌ Gagal membuka webcam!")
        print("➡️ Berikan izin kamera terlebih dahulu di OS/browser.")
        exit()

    print("✅ Webcam berhasil dibuka — mulai deteksi...")

    while True:
        ret, frame = cam.read()

        if not ret:
            print("❌ Tidak bisa membaca frame dari kamera.")
            break

        # YOLO detect
        results = model(frame)
        detections = results.xyxy[0].numpy()

        # Draw boxes
        if len(detections) > 0:
            draw_boxes(frame, detections)

        cv2.imshow("Waste Detection", frame)

        # stop = tekan q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Deteksi dihentikan.")
            break

    cam.release()
    cv2.destroyAllWindows()
