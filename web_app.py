import json
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from torchvision import transforms

from drowsiness_model import DrowsinessCNN


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAME_BYTES = 2 * 1024 * 1024

with (ROOT / "class_labels.json").open(encoding="utf-8") as labels_file:
    class_info = json.load(labels_file)

CLASS_LABELS = class_info["class_names"]
IMG_SIZE = class_info.get("img_size", 145)
model = DrowsinessCNN(num_classes=len(CLASS_LABELS)).to(DEVICE)
model.load_state_dict(
    torch.load(
        ROOT / "driver_drowsiness_model.pth",
        map_location=DEVICE,
        weights_only=True,
    )
)
model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("OpenCV Haar Cascade files could not be loaded.")

inference_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = Flask(__name__)


def predict_region(image):
    if image is None or image.size == 0:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = inference_transform(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    return {
        "label": CLASS_LABELS[class_index.item()],
        "confidence": round(confidence.item() * 100, 1),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/detect")
def detect():
    if request.content_length is not None and request.content_length > MAX_FRAME_BYTES:
        return jsonify(error="Frame is too large."), 413

    encoded_frame = request.get_data()
    if not encoded_frame or len(encoded_frame) > MAX_FRAME_BYTES:
        return jsonify(error="Send a valid image frame under 2 MB."), 400

    frame = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="The uploaded frame could not be decoded."), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
    )
    if len(faces) == 0:
        return jsonify(face_detected=False, eyes=None, mouth=None)

    fx, fy, fw, fh = faces[0]
    face = frame[fy : fy + fh, fx : fx + fw]
    face_gray = gray[fy : fy + fh, fx : fx + fw]
    upper_height = int(fh * 0.6)
    eyes = eye_cascade.detectMultiScale(
        face_gray[:upper_height],
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
    )

    eye_result = None
    if len(eyes) > 0:
        ex, ey, ew, eh = eyes[0]
        eye_result = predict_region(
            face[ey : ey + eh, ex : ex + ew]
        )
        if eye_result and eye_result["label"] not in ("Open", "Closed"):
            eye_result = {"label": "Open", "confidence": 50.0}

    mouth_top = int(fh * 0.6)
    mouth_result = predict_region(face[mouth_top:fh, :])
    if mouth_result and mouth_result["label"] not in ("yawn", "no_yawn"):
        mouth_result = {"label": "no_yawn", "confidence": 50.0}

    return jsonify(face_detected=True, eyes=eye_result, mouth=mouth_result)


if __name__ == "__main__":
    print(f"Model ready on {DEVICE}. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
