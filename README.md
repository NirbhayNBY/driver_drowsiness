# Driver Drowsiness Detection & Alert System

A real-time **Driver Drowsiness Detection System** built with Python, using a **CNN (Convolutional Neural Network)** trained on eye and mouth images to detect if a driver is falling asleep or yawning -- and alerts them with an alarm.

---

## Features

| Feature | Description |
|---------|-------------|
| **CNN Model** | Custom 4-layer CNN trained on 2,900+ images |
| **4-Class Detection** | Closed eyes, Open eyes, Yawn, No Yawn |
| **Real-Time Camera** | Uses webcam with OpenCV for live detection |
| **Sleeping Alert** | Alarm + visual warning when eyes stay closed |
| **Yawn Warning** | Beep + on-screen alert when yawning detected |
| **Confidence Score** | Shows prediction label + confidence % on screen |
| **Alarm Sound** | Windows beep alarm (no external files needed) |

---

## Project Structure

```
driver_drowsiness/
|
+-- archive (1)/dataset_new/      # Dataset
|   +-- train/
|   |   +-- Closed/      (617 images)
|   |   +-- Open/        (617 images)
|   |   +-- no_yawn/     (616 images)
|   |   +-- yawn/        (617 images)
|   +-- test/
|       +-- Closed/      (109 images)
|       +-- Open/        (109 images)
|       +-- no_yawn/     (109 images)
|       +-- yawn/        (106 images)
|
+-- train_model.py                # Model training script (PyTorch)
+-- detect_drowsiness.py          # Real-time detection script
+-- requirements.txt              # Python dependencies
+-- README.md                     # This file
|
+-- driver_drowsiness_model.pth   # Trained model (after training)
+-- class_labels.json             # Class names mapping (after training)
+-- training_results.png          # Training plots (after training)
```

---

## Requirements

- **Python** 3.8 or higher (tested on 3.14)
- **Webcam** (built-in or external)
- **Windows OS** (for `winsound` alarm)

### Python Packages

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch - CNN model building & training |
| `torchvision` | Image transforms & dataset loading |
| `opencv-python` | Webcam capture & face detection |
| `numpy` | Numerical operations |
| `matplotlib` | Training result plots |
| `Pillow` | Image processing |

---

## How to Run -- Step by Step

### Step 1: Install Python Dependencies

Open a terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

This trains the CNN on the dataset and saves the model file:

```bash
python train_model.py
```

**What happens:**
- Loads 2,467 training images from the dataset
- Trains a 4-layer CNN with data augmentation
- Evaluates on 433 test images
- Saves the best model as `driver_drowsiness_model.pth`
- Saves class mapping as `class_labels.json`
- Saves accuracy/loss plots as `training_results.png`

Training time: ~5-15 minutes (depends on your hardware, faster with GPU)

### Step 3: Run Real-Time Detection

After training is complete, start the drowsiness detection:

```bash
python detect_drowsiness.py
```

**What happens:**
- Opens your webcam
- Detects your face using Haar Cascade
- Extracts eye and mouth regions
- Uses the trained CNN model to predict eye/mouth state
- Shows prediction label and confidence on screen
- Triggers alarm if drowsiness is detected

### Step 4: Test the Alerts

| Action | Expected Alert |
|--------|---------------|
| Close your eyes for ~1 second | **"DRIVER IS SLEEPING! WAKE UP!"** + loud alarm |
| Yawn / open mouth wide | **"DRIVER IS DROWSY!"** + warning beep |
| Keep eyes open normally | **"Status: ACTIVE"** |

### Step 5: Quit

Press **`q`** on the keyboard to stop the detection and close the camera.

---

## How It Works

```
+----------------+     +-----------------+     +----------------+
|  Webcam        |---->|  Face Detect    |---->|  Extract       |
|  Capture       |     |  (Haar Cascade) |     |  Eye + Mouth   |
+----------------+     +-----------------+     +-------+--------+
                                                       |
                                               +-------v--------+
                                               |  CNN Model     |
                                               |  Prediction    |
                                               |  (4 classes)   |
                                               +-------+--------+
                                                       |
                        +---------------------+--------+--------+
                        |                     |                  |
              +---------v------+   +----------v-----+  +--------v------+
              | Eyes Closed    |   | Yawn Detected  |  | All Normal    |
              | > 15 frames    |   | > 5 frames     |  |               |
              | ALARM!         |   | WARNING!       |  | ACTIVE        |
              +----------------+   +----------------+  +---------------+
```

### CNN Model Architecture

```
Input (145x145x3 RGB)
    |
    +-- Conv2D(32) + BatchNorm + ReLU + MaxPool
    +-- Conv2D(64) + BatchNorm + ReLU + MaxPool
    +-- Conv2D(128) + BatchNorm + ReLU + MaxPool
    +-- Conv2D(256) + BatchNorm + ReLU + MaxPool
    |
    +-- Flatten
    +-- Linear(512) + ReLU + Dropout(0.5)
    +-- Linear(256) + ReLU + Dropout(0.3)
    +-- Linear(4) -> [Closed, Open, no_yawn, yawn]
```

---

## Configuration

You can adjust these values in `detect_drowsiness.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EYE_CLOSED_THRESHOLD` | 15 frames | Frames before sleep alarm triggers |
| `YAWN_THRESHOLD` | 5 frames | Frames before yawn warning triggers |
| `ALARM_FREQ` | 2500 Hz | Alarm sound frequency |
| `ALARM_DURATION` | 1000 ms | Alarm sound duration |
| `PROCESS_EVERY_N` | 2 | Process every Nth frame (for speed) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot access webcam` | Check webcam is connected and not used by another app |
| `Model not found` | Run `python train_model.py` first |
| `Low accuracy` | Try increasing `EPOCHS` in `train_model.py` |
| `Slow detection` | Increase `PROCESS_EVERY_N` or reduce `IMG_SIZE` |
| `No alarm sound` | Make sure system volume is on; `winsound` only works on Windows |
| `tensorflow not found` | This project uses **PyTorch**, not TensorFlow. Run `pip install -r requirements.txt` |

---

## Notes

- This project uses **only ML-based detection** -- no manual EAR formula or MediaPipe
- The Haar Cascade is used only for **locating** the face/eyes -- classification is done by the **CNN model**
- The alarm uses `winsound.Beep()` which is built into Windows -- no extra files needed
- For best results, ensure good lighting and face the camera directly
- Uses **PyTorch** (works with Python 3.8 - 3.14+)

---

## Tech Stack

- **Python 3.x**
- **PyTorch / TorchVision** -- Deep Learning
- **OpenCV** -- Computer Vision
- **NumPy** -- Numerical Computing
- **Matplotlib** -- Visualization
- **winsound** -- Alarm (Windows built-in)
