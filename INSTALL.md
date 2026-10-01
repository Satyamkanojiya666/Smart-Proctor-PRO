# 📦 Installation Guide — AI Exam Proctoring System

> Complete setup instructions for Windows, macOS, and Linux.
> GitHub: https://github.com/Satyamkanojiya666

---

## ✅ System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 + |
| RAM | 4 GB | 8 GB |
| CPU | Dual-core | Quad-core |
| Webcam | 480p | 720p or higher |
| Disk Space | 150 MB | 300 MB |
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Latest |

> ⚠️ **GPU is NOT required.** Everything runs on CPU out of the box.

---

## 🪟 Windows — Step by Step

### Step 1 — Install Python
Download Python 3.10+ from https://python.org/downloads  
During install, check ✅ **"Add Python to PATH"**

Verify:
```cmd
python --version
pip --version
```

### Step 2 — (Recommended) Create a Virtual Environment
```cmd
python -m venv proctor_env
proctor_env\Scripts\activate
```

### Step 3 — Install Dependencies
```cmd
pip install -r requirements.txt
```

Or install manually:
```cmd
pip install opencv-python mediapipe numpy
```

### Step 4 — Run
```cmd
python exam_proctor_elite.py
```

---

## 🍎 macOS — Step by Step

### Step 1 — Install Python (via Homebrew recommended)
```bash
brew install python@3.10
```

Or download from https://python.org/downloads

### Step 2 — (Recommended) Create a Virtual Environment
```bash
python3 -m venv proctor_env
source proctor_env/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy
```

### Step 4 — Allow Camera Access
Go to **System Settings → Privacy & Security → Camera**  
Enable access for **Terminal** (or your IDE).

### Step 5 — Run
```bash
python3 exam_proctor_elite.py
```

---

## 🐧 Linux (Ubuntu / Debian) — Step by Step

### Step 1 — Install Python & pip
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

### Step 2 — Install system dependencies (camera & audio)
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0 alsa-utils -y
```

### Step 3 — (Recommended) Create a Virtual Environment
```bash
python3 -m venv proctor_env
source proctor_env/bin/activate
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy
```

### Step 5 — Run
```bash
python3 exam_proctor_elite.py
```

---

## 📥 First Run — Auto Downloads

On the very first run, the system will automatically download:

| File | Size | Purpose |
|---|---|---|
| `face_landmarker.task` | ~30 MB | MediaPipe face landmark model |
| `yolov4-tiny.cfg` | ~8 KB | YOLO network config |
| `yolov4-tiny.weights` | ~24 MB | YOLO pre-trained weights |
| `coco.names` | ~1 KB | COCO class labels |

> ✅ After first run, the system works **fully offline**.

---

## 🔁 Updating Dependencies

To upgrade all packages to their latest compatible versions:

```bash
pip install --upgrade opencv-python mediapipe numpy
```

---

## 🐍 Using a Specific Python Version

If you have multiple Python versions installed:

```bash
# Check available versions
python3 --version
python3.10 --version

# Create venv with specific version
python3.10 -m venv proctor_env
source proctor_env/bin/activate   # Linux/macOS
proctor_env\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## ❗ Common Errors & Fixes

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'mediapipe'`
```bash
pip install mediapipe
```

### Camera not opening / black screen
- Make sure no other app (Zoom, Teams) is using the webcam
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`
- On Linux: check `ls /dev/video*` to confirm camera is detected

### `libGL.so.1: cannot open shared object file` (Linux)
```bash
sudo apt install libgl1-mesa-glx -y
```

### `aplay` not found / no alarm sound (Linux)
```bash
sudo apt install alsa-utils -y
```

### Slow FPS / lag
- Close other heavy applications
- Reduce webcam resolution in code: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
- YOLO runs every 5 frames by default — increase `SKIP_FRAMES` to reduce load

### Download fails on first run
- Check your internet connection
- Re-run the script — downloads resume automatically
- Or manually place model files in the same folder as the script

---

## 📁 Expected File Structure After Setup

```
ProctorAI/
├── exam_proctor_elite.py       ← main script
├── requirements.txt            ← this file
├── INSTALL.md                  ← install guide
├── README.md                   ← project overview
├── face_landmarker.task        ← auto-downloaded (~30 MB)
├── yolov4-tiny.cfg             ← auto-downloaded
├── yolov4-tiny.weights         ← auto-downloaded (~24 MB)
├── coco.names                  ← auto-downloaded
├── alarm.wav                   ← auto-generated
├── logs/
│   ├── events_YYYYMMDD_HHMMSS.csv
│   └── report_YYYYMMDD_HHMMSS.json
└── screenshots/
    └── *.jpg
```

---

## 🧪 Quick Test (verify install)

Run this in Python to confirm all packages are installed correctly:

```python
import cv2
import mediapipe
import numpy

print("OpenCV  :", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("NumPy   :", numpy.__version__)
print("✅ All dependencies installed successfully!")
```

---

*For issues or contributions, visit: https://github.com/Satyamkanojiya666*
