# 🎓 AI Exam Proctoring System — Elite Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![YOLOv4](https://img.shields.io/badge/YOLOv4--tiny-Device%20Detection-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Version](https://img.shields.io/badge/Version-2.0.0--elite-purple)

**Real-time AI-powered exam proctoring using facial landmark analysis, 3D head pose estimation, iris gaze tracking, and electronic device detection.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Controls](#️-controls) • [Output](#-output) • [FAQ](#-faq)

</div>

---

## 📸 Overview

This system uses your webcam and computer vision models to monitor exam candidates in real time. It detects suspicious behaviors such as looking sideways, head turning, extended eye closure, multiple faces in frame, and electronic devices — then logs events and generates a detailed session report.

```
Engine  : MediaPipe 0.10+ FaceLandmarker (478 landmarks) + YOLOv4-tiny
Tracking: 3D Head Pose · Iris Gaze · EAR Blink · Face Distance · Device Scan
```

---

## ✨ Features

| Category | Detection |
|---|---|
| 👁️ **Gaze Tracking** | Eye left / right / up / down / center via iris landmarks |
| 🗣️ **Head Pose (3D)** | Yaw (left/right) · Pitch (up/down) · Roll — via solvePnP |
| 😶 **Blink Analysis** | EAR-based blink rate monitoring, extended closure alert |
| 👥 **Multi-Face** | Tracks and IDs multiple faces, flags intruder entry |
| 📵 **Device Detection** | Phone · Laptop · Mouse · Keyboard · Remote via YOLOv4-tiny |
| 📖 **Suspicious Items** | Books/Notes · Smartwatch flagged as warnings |
| 📏 **Distance Check** | Alerts if candidate is too close or too far from camera |
| 🏃 **Fidget Detection** | Head movement variance analysis for restlessness |
| 🔊 **Alarm** | Audio alert on every confirmed cheat event |
| 📊 **Reports** | JSON risk report + timestamped CSV event log per session |
| 📸 **Screenshots** | Manual screenshot capture during live session |

---

## 🚦 Behavior Rules

```
✅ ALLOWED   →  Eye Up / Down / Center, small natural head movement
❌ CHEAT     →  Eye Left/Right, Head Left/Right, Head Tilt Up,
                Face Tilt Down, Extended Eyes Closed, No Face Detected,
                Multiple Faces, New Person Entering Frame,
                📱 Phone / 💻 Laptop / ⌨️ Keyboard Detected
⚠  WARN      →  Abnormal blink rate, Too close/far, Fidgeting,
                Book/Notes visible, Smartwatch detected
```

### Priority Order (higher wins if multiple triggers fire simultaneously)

```
0 → Electronic Device Detected  🚨  (HIGHEST PRIORITY)
1 → Multiple Faces Detected     🚨
2 → Face Not Detected           🚨
3 → Head Looking Left / Right   ❌
4 → Eye Gaze Left / Right       ❌
5 → Head Tilt Up / Face Down    ❌
6 → Blink / Distance / Fidget   ⚠
```

---

## 🛠 Installation

### Requirements

- Python 3.8 or higher
- A working webcam
- ~60 MB free disk space (models downloaded on first run)

### Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

> **Note:** No manual model downloads required. On first run, the system automatically downloads:
> - `face_landmarker.task` (~30 MB) — MediaPipe face model
> - `yolov4-tiny.cfg`, `yolov4-tiny.weights`, `coco.names` (~25 MB total) — device detection

---

## 🚀 Usage

```bash
python exam_proctor_elite.py
```

The window will open your webcam feed with the full proctoring overlay active. All events are logged in real time to `logs/`.

---

## 🧠 How It Works

### Face Landmark Detection
MediaPipe's `FaceLandmarker` model detects **478 facial landmarks** per frame including precise iris positions, enabling millimeter-level gaze estimation without any additional eye-tracking hardware.

### 3D Head Pose Estimation
Six key landmarks are matched to a 3D face model using OpenCV's `solvePnP`. The rotation matrix is decomposed into **Pitch** (up/down), **Yaw** (left/right), and **Roll** (tilt) angles. Thresholds beyond ±20° trigger cheat flags.

### Iris Gaze Estimation
Iris center is calculated relative to eye corners for both eyes. The horizontal ratio determines left/right gaze; the vertical ratio determines up/down. Both eyes are averaged for stability.

### EAR Blink Detection
The **Eye Aspect Ratio (EAR)** is computed from six eye landmarks. A value below 0.20 for 3+ consecutive frames counts as a blink. Blink rate outside the 6–40 blinks/min range is flagged as suspicious.

### Electronic Device Detection (YOLOv4-tiny)
YOLOv4-tiny runs on the CPU, scanning every 5th frame for COCO-class objects. Detected devices are classified as either cheat-level (phone, laptop, keyboard, mouse, remote) or warn-level (book, smartwatch).

### Multi-Face Tracking
Haar cascade detects all faces; a centroid-based tracker assigns persistent IDs across frames. Any face ID > 1 is flagged as an intruder with a `NEW PERSON ENTERED` event.

---

## 🕹️ Controls

| Key | Action |
|-----|--------|
| `ESC` | End session and generate report |
| `D` | Toggle iris debug panel |
| `R` | Reset cheat counter |
| `S` | Save screenshot to `screenshots/` |

---

## 📁 Output

```
project/
├── exam_proctor_elite.py
├── face_landmarker.task          ← auto-downloaded
├── yolov4-tiny.cfg               ← auto-downloaded
├── yolov4-tiny.weights           ← auto-downloaded
├── coco.names                    ← auto-downloaded
├── alarm.wav                     ← auto-generated
├── logs/
│   ├── events_YYYYMMDD_HHMMSS.csv    ← timestamped event log
│   └── report_YYYYMMDD_HHMMSS.json   ← session risk report
└── screenshots/
    └── YYYYMMDD_HHMMSS_shot001.jpg
```

### Sample JSON Report

```json
{
  "session_id": "20240427_143012",
  "duration": "12m 30s",
  "total_cheat_events": 7,
  "risk_level": "Medium ⚠⚠",
  "stats": {
    "head_left": 3,
    "eye_gaze_right": 2,
    "device_phone": 1,
    "multiple_faces": 1
  },
  "events": [...]
}
```

### Risk Levels

| Rate (events/min) | Risk Level |
|---|---|
| 0 | ✅ Clean |
| < 2 | ⚠ Low |
| 2 – 5 | ⚠⚠ Medium |
| > 5 | 🚨 High |

---

## ⚙️ Configuration

You can tune detection sensitivity by editing these constants at the top of the script:

```python
# Head pose thresholds (degrees)
YAW_THRESH   = 20   # left/right head turn
PITCH_THRESH = 20   # up/down head tilt

# Blink detection
EARBlinkDetector.EAR_THRESH = 0.20   # eye closure sensitivity
EARBlinkDetector.CONSEC     = 3      # frames before counting as blink

# Device detection
ElectronicDeviceDetector.CONF_THRESH = 0.40   # YOLO confidence threshold
ElectronicDeviceDetector.SKIP_FRAMES = 5      # run YOLO every N frames

# Movement / fidgeting
HeadMovementTracker.fidget_thresh = 32   # variance threshold
```

---

## 📋 Requirements Summary

```
opencv-python >= 4.5
mediapipe     >= 0.10
numpy         >= 1.21
```

---

## ❓ FAQ

**Does this work without internet?**
After first run (model download), yes — fully offline.

**Does it work on Mac / Linux?**
Yes. Alarm uses `afplay` on macOS and `aplay` on Linux. Camera backend falls back gracefully.

**Can I use a USB webcam instead of built-in?**
Yes, change `cv2.VideoCapture(0, ...)` index from `0` to `1` or `2` for external cameras.

**How accurate is device detection?**
YOLOv4-tiny achieves good real-time performance on CPU. For higher accuracy at the cost of speed, swap to full YOLOv4 or YOLOv8.

**Can I add more prohibited items?**
Yes — add COCO class IDs to `DEVICE_CLASS_IDS` and include them in `CHEAT_DEVICE_IDS` or `WARN_DEVICE_IDS`.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) — Face Landmark model
- [AlexeyAB / Darknet](https://github.com/AlexeyAB/darknet) — YOLOv4-tiny weights
- [OpenCV](https://opencv.org) — Computer vision backbone

---

<div align="center">
Made with ❤️ by <a href="https://github.com/YasirShaikh03">Satyam Kanojiya</a>
</div>
