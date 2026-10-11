"""
=============================================================
 Driver Drowsiness Detection -- Real-Time Detection Script
=============================================================
 Uses the trained CNN model to detect driver drowsiness
 through the webcam in real-time.

 What it does:
   1. Captures video from webcam
   2. Detects face using Haar Cascade
   3. Extracts eye region -> predicts Open / Closed
   4. Extracts mouth region -> predicts yawn / no_yawn
   5. If eyes closed for too long -> ALARM (sleeping)
   6. If yawning detected -> WARNING (drowsy)

 Prerequisites:
   - Trained model file: driver_drowsiness_model.pth
   - Class labels file: class_labels.json
   - Run train_model.py first if these files don't exist

 Controls:
   - Press 'q' to quit the application

 Usage:
   python detect_drowsiness.py
=============================================================
"""

import os
import sys
import json
import cv2
import numpy as np
import threading
import time

import torch
from torchvision import transforms
from drowsiness_model import DrowsinessCNN

import winsound
ALARM_SOUND = r"COMCell_Iphone ringtone alarm (ID 0451)_BigSoundBank.com.wav"

# =====================================================
# 1. CONFIGURATION -- Tweak these values if needed
# =====================================================

# Path to the trained model and labels
MODEL_PATH = "driver_drowsiness_model.pth"
LABELS_PATH = "class_labels.json"

# Image size must match the training input size
IMG_SIZE = 100

# How many consecutive frames of closed eyes before alarm?
# At ~15 FPS, 15 frames = approx 1 second of continuous closure
EYE_CLOSED_THRESHOLD = 15

# How many consecutive yawn frames before warning?
YAWN_THRESHOLD = 5

# Alarm sound settings (Windows beep)
# Colours in BGR format (OpenCV uses BGR, not RGB)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

# Device for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 3. LOAD MODEL AND HAAR CASCADES
# =====================================================

print("=" * 60)
print("  DRIVER DROWSINESS DETECTION -- REAL-TIME MODE")
print("=" * 60)
print(f"   Device: {DEVICE}")

# --- Load class labels ---
print("\n[1/4] Loading class labels...")
if not os.path.isfile(LABELS_PATH):
    print(f"      ERROR: Labels file not found: {LABELS_PATH}")
    print("      Please run  python train_model.py  first!")
    sys.exit(1)

with open(LABELS_PATH, 'r') as f:
    class_info = json.load(f)

CLASS_LABELS = class_info['class_names']  # ['Closed', 'Open', 'no_yawn', 'yawn']
IMG_SIZE = class_info.get('img_size', 145)
print(f"      Classes: {CLASS_LABELS}")
print(f"      Image size: {IMG_SIZE}x{IMG_SIZE}")

# --- Load the trained CNN model ---
print("[2/4] Loading trained model...")
if not os.path.isfile(MODEL_PATH):
    print(f"      ERROR: Model file not found: {MODEL_PATH}")
    print("      Please run  python train_model.py  first!")
    sys.exit(1)

try:
    model = DrowsinessCNN(num_classes=len(CLASS_LABELS)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()  # Set to evaluation mode (disables dropout)
    print(f"      Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"      ERROR loading model: {e}")
    print("      The model file may be corrupted. Re-run train_model.py")
    sys.exit(1)

# --- Load Haar Cascade classifiers ---
print("[3/4] Loading Haar Cascades...")

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

if not os.path.isfile(face_cascade_path):
    print(f"      ERROR: Face cascade not found: {face_cascade_path}")
    sys.exit(1)
if not os.path.isfile(eye_cascade_path):
    print(f"      ERROR: Eye cascade not found: {eye_cascade_path}")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty() or eye_cascade.empty():
    print("      ERROR: Could not load Haar Cascades!")
    sys.exit(1)

print("      Haar Cascades loaded successfully")


# =====================================================
# 4. IMAGE PREPROCESSING PIPELINE
# =====================================================

# Same normalization as during training
inference_transform = transforms.Compose([
    transforms.ToPILImage(),                       # numpy -> PIL
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # Resize
    transforms.ToTensor(),                         # PIL -> tensor (0-1)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def preprocess_for_model(image_bgr):
    """
    Preprocess an OpenCV BGR image for the PyTorch model.
    Returns a tensor ready for model input, or None if image is invalid.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    try:
        # Convert BGR (OpenCV) to RGB (PyTorch expects RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Apply transforms: resize, to tensor, normalize
        tensor = inference_transform(image_rgb)
        # Add batch dimension: (C,H,W) -> (1,C,H,W)
        tensor = tensor.unsqueeze(0).to(DEVICE)
        return tensor
    except Exception:
        return None


def predict_class(model, image_tensor):
    """
    Run model prediction on a preprocessed image tensor.
    Returns (class_name, confidence_percentage).
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, class_idx = torch.max(probs, 0)
        class_name = CLASS_LABELS[class_idx.item()]
        conf_percent = confidence.item() * 100
    return class_name, conf_percent


# Alarm state (use list for mutability in closures)
alarm_state = [False]


import os

alarm_state = [False]

def play_alarm_sound():
    if alarm_state[0]:
        return

    def _play():
        alarm_state[0] = True
        try:
            winsound.PlaySound(ALARM_SOUND, winsound.SND_FILENAME)
        except Exception as e:
            print("Audio Error:", e)
        finally:
            alarm_state[0] = False

    threading.Thread(target=_play, daemon=True).start()

def draw_text_with_bg(frame, text, pos, bg_color,
                      font_scale=0.7, thickness=2):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(th + 10, min(y, frame_h - 1))
    cv2.rectangle(frame, (x, y - th - 10),
                  (x + tw + 10, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x + 5, y - 5),
                font, font_scale, WHITE, thickness)


# =====================================================
# 6. START WEBCAM
# =====================================================

print("[4/4] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("      ERROR: Cannot access webcam!")
    print("      Make sure your webcam is connected and not in use.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("      Webcam started successfully")
print("\n" + "=" * 60)
print("  DETECTION RUNNING -- Press 'q' to quit")
print("=" * 60 + "\n")


# =====================================================
# 7. MAIN DETECTION LOOP
# =====================================================

# Counters for consecutive detections
eye_closed_counter = 0
yawn_counter = 0

# Current status for display
eye_status = "Detecting..."
eye_conf = 0.0
mouth_status = "Detecting..."
mouth_conf = 0.0

# Frame skip for performance
frame_count = 0
PROCESS_EVERY_N = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam")
        break

    # Mirror for natural feel
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    frame_count += 1

    process_this_frame = (frame_count % PROCESS_EVERY_N == 0)

    if process_this_frame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
        )

        face_processed = False

        for (fx, fy, fw, fh) in faces:
            if face_processed:
                break  # Only process first face
            face_processed = True

            # Clamp to frame bounds
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, frame_w - fx)
            fh = min(fh, frame_h - fy)
            if fw < 50 or fh < 50:
                continue

            # Draw face box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), GREEN, 2)
            cv2.putText(frame, "Face", (fx, fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

            # Face ROIs
            face_color = frame[fy:fy + fh, fx:fx + fw]
            face_gray = gray[fy:fy + fh, fx:fx + fw]

            # ===== EYE DETECTION =====
            upper_h = int(fh * 0.6)
            if upper_h < 20:
                continue

            upper_face_gray = face_gray[0:upper_h, :]
            upper_face_color = face_color[0:upper_h, :]

            eyes = eye_cascade.detectMultiScale(
                upper_face_gray,
                scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )

            eye_predicted = False

            if len(eyes) > 0:
                (ex, ey, ew, eh) = eyes[0]
                # Clamp eye ROI
                ex = max(0, ex)
                ey = max(0, ey)
                ew = min(ew, upper_face_color.shape[1] - ex)
                eh = min(eh, upper_face_color.shape[0] - ey)

                eye_roi = upper_face_color[ey:ey + eh, ex:ex + ew]
                eye_input = preprocess_for_model(eye_roi)

                if eye_input is not None:
                    eye_predicted = True
                    eye_label, eye_confidence = predict_class(model, eye_input)

                    if eye_label == 'Closed':
                        eye_status = "Closed"
                        eye_conf = eye_confidence
                        eye_closed_counter += 1
                    elif eye_label == 'Open':
                        eye_status = "Open"
                        eye_conf = eye_confidence
                        eye_closed_counter = 0
                    else:
                        # Model gave mouth class for eye region
                        # Eye was detected by Haar, so assume open
                        eye_status = "Open"
                        eye_conf = 50.0
                        eye_closed_counter = 0

                    # Draw eye box
                    eye_color = RED if eye_status == "Closed" else YELLOW
                    cv2.rectangle(frame,
                                  (fx + ex, fy + ey),
                                  (fx + ex + ew, fy + ey + eh),
                                  eye_color, 2)

            if not eye_predicted:
                # No eyes found -- might be closed
                eye_closed_counter += 1

            # ===== MOUTH / YAWN DETECTION =====
            mouth_top = int(fh * 0.6)
            mouth_roi = face_color[mouth_top:fh, :]
            mouth_input = preprocess_for_model(mouth_roi)

            if mouth_input is not None:
                mouth_label, mouth_confidence = predict_class(model, mouth_input)

                if mouth_label == 'yawn':
                    mouth_status = "yawn"
                    mouth_conf = mouth_confidence
                    yawn_counter += 1
                elif mouth_label == 'no_yawn':
                    mouth_status = "no_yawn"
                    mouth_conf = mouth_confidence
                    yawn_counter = 0
                else:
                    mouth_status = "no_yawn"
                    mouth_conf = 50.0
                    yawn_counter = 0

                mouth_color = ORANGE if mouth_status == "yawn" else GREEN
                cv2.rectangle(frame,
                              (fx, fy + mouth_top),
                              (fx + fw, fy + fh),
                              mouth_color, 2)

        # No face -> slowly decrease counters
        if not face_processed:
            eye_closed_counter = max(0, eye_closed_counter - 1)
            yawn_counter = max(0, yawn_counter - 1)

    # =====================================================
    # 8. ALERT LOGIC
    # =====================================================

    if eye_closed_counter >= EYE_CLOSED_THRESHOLD:
        draw_text_with_bg(frame, "DRIVER IS SLEEPING! WAKE UP!",
                          (10, 60), RED, font_scale=0.8, thickness=2)
        play_alarm_sound()

    elif yawn_counter >= YAWN_THRESHOLD:
        draw_text_with_bg(frame, "DRIVER IS DROWSY!",
                          (10, 60), ORANGE, font_scale=0.8, thickness=2)
        play_alarm_sound()

    else:
        draw_text_with_bg(frame, "Status: ACTIVE", (10, 60),
                          (0, 130, 0), font_scale=0.7)

    # =====================================================
    # 9. DRAW INFO ON SCREEN
    # =====================================================

    # Title bar
    cv2.rectangle(frame, (0, 0), (frame_w, 30), BLACK, -1)
    cv2.putText(frame, "Driver Drowsiness Detection System",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    # Eye status
    e_color = RED if eye_status == "Closed" else GREEN
    cv2.putText(frame, f"Eyes: {eye_status} ({eye_conf:.1f}%)",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, e_color, 2)

    # Mouth status
    m_color = ORANGE if mouth_status == "yawn" else GREEN
    cv2.putText(frame, f"Mouth: {mouth_status} ({mouth_conf:.1f}%)",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 2)

    # Counters at bottom
    counter_y = frame_h - 15
    cv2.putText(frame,
                f"Eye Closed: {eye_closed_counter}/{EYE_CLOSED_THRESHOLD}"
                f"  |  Yawn: {yawn_counter}/{YAWN_THRESHOLD}",
                (10, counter_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    cv2.putText(frame, "Press 'q' to quit",
                (frame_w - 160, counter_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # Show frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =====================================================
# 10. CLEANUP
# =====================================================

print("\nStopping detection...")
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")
