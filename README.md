<div align="center">

# Smart Proctor Pro

**Real-time exam monitoring using facial landmark analysis, 3D head pose estimation, iris gaze tracking, and electronic device detection.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![YOLOv4-tiny](https://img.shields.io/badge/YOLOv4--tiny-Device%20Detection-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Version](https://img.shields.io/badge/Version-2.0.0-purple)

[Features](#features) · [Installation](#installation) · [Usage](#usage) · [How It Works](#how-it-works) · [Configuration](#configuration) · [Output](#output) · [FAQ](#faq)

</div>

---

## Overview

Smart Proctor Pro uses a standard webcam and computer vision models to monitor exam candidates in real time. It detects behaviors commonly associated with academic dishonesty, such as looking away from the screen, turning the head, prolonged eye closure, additional people in frame, and prohibited electronic devices. Every event is logged, and a risk report is generated at the end of each session.

| Component | Technology |
|---|---|
| Face analysis | MediaPipe FaceLandmarker (478 landmarks) |
| Object detection | YOLOv4-tiny (CPU inference) |
| Tracking | 3D head pose, iris gaze, EAR blink, face distance, device scan |
| Runtime | Python 3.8+, OpenCV, NumPy |

---

## Features

| Category | Capability |
|---|---|
| **Gaze tracking** | Classifies gaze as left, right, up, down, or center using iris landmarks |
| **3D head pose** | Estimates yaw, pitch, and roll via `solvePnP` |
| **Blink analysis** | EAR-based blink-rate monitoring and extended-closure alerts |
| **Multi-face tracking** | Assigns persistent IDs to faces and flags new persons entering the frame |
| **Device detection** | Detects phones, laptops, mice, keyboards, and remotes with YOLOv4-tiny |
| **Suspicious items** | Flags books/notes and smartwatches as warnings |
| **Distance check** | Alerts when the candidate is too close to or too far from the camera |
| **Fidget detection** | Measures head-movement variance to identify restlessness |
| **Audio alarm** | Plays an alert on every confirmed violation |
| **Reporting** | Per-session JSON risk report and timestamped CSV event log |
| **Screenshots** | Manual capture during a live session |

---

## Behavior Rules

| Level | Behaviors |
|---|---|
| **Allowed** | Eyes up, down, or center; small natural head movements |
| **Violation** | Eye gaze left/right; head turned left/right; head tilted up; face tilted down; extended eye closure; no face detected; multiple faces; new person entering frame; phone, laptop, or keyboard detected |
| **Warning** | Abnormal blink rate; too close or too far from camera; fidgeting; book or notes visible; smartwatch detected |

### Priority Order

When several triggers fire at once, the highest-priority event takes precedence.

| Priority | Event | Severity |
|:---:|---|---|
| 0 | Electronic device detected | Critical (highest) |
| 1 | Multiple faces detected | Critical |
| 2 | Face not detected | Critical |
| 3 | Head turned left / right | Violation |
| 4 | Eye gaze left / right | Violation |
| 5 | Head tilted up / face down | Violation |
| 6 | Blink rate / distance / fidgeting | Warning |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam
- Approximately 60 MB of free disk space for models

### Setup

```bash
git clone <repository-url>
cd <repository-folder>
pip install opencv-python mediapipe numpy
```

No manual model downloads are required. On first run the system fetches:

| File | Size | Purpose |
|---|---|---|
| `face_landmarker.task` | ~30 MB | MediaPipe face model |
| `yolov4-tiny.cfg`, `yolov4-tiny.weights`, `coco.names` | ~25 MB total | Device detection |

An internet connection is needed only for this first run.

---

## Usage

```bash
python smart_proctor_pro.py
```

A window opens showing the webcam feed with the proctoring overlay. Events are written to `logs/` in real time.

### Controls

| Key | Action |
|---|---|
| `ESC` | End the session and generate the report |
| `D` | Toggle the iris debug panel |
| `R` | Reset the violation counter |
| `S` | Save a screenshot to `screenshots/` |

---

## How It Works

### Face Landmark Detection
MediaPipe's FaceLandmarker returns 478 landmarks per frame, including iris positions, which allows gaze estimation without dedicated eye-tracking hardware.

### 3D Head Pose Estimation
Six key landmarks are matched against a 3D face model with OpenCV's `solvePnP`. The resulting rotation matrix is decomposed into pitch, yaw, and roll. Angles beyond ±20° raise a violation.

### Iris Gaze Estimation
The iris center is measured relative to each eye's corners. The horizontal ratio determines left/right gaze and the vertical ratio determines up/down. Values from both eyes are averaged for stability.

### Blink Detection (EAR)
The Eye Aspect Ratio is computed from six eye landmarks. A value below 0.20 for three or more consecutive frames counts as a blink. A rate outside 6–40 blinks per minute is flagged as suspicious.

### Electronic Device Detection
YOLOv4-tiny runs on the CPU and scans every 5th frame for COCO-class objects. Detections are categorized as violation-level (phone, laptop, keyboard, mouse, remote) or warning-level (book, smartwatch).

### Multi-Face Tracking
A Haar cascade detects all faces in the frame, and a centroid-based tracker assigns persistent IDs. Any face with an ID greater than 1 is treated as an intruder and logged as `NEW PERSON ENTERED`.

---

## Configuration

Detection sensitivity can be tuned through constants at the top of the script.

```python
# Head pose thresholds (degrees)
YAW_THRESH   = 20   # left/right head turn
PITCH_THRESH = 20   # up/down head tilt

# Blink detection
EARBlinkDetector.EAR_THRESH = 0.20   # eye-closure sensitivity
EARBlinkDetector.CONSEC     = 3      # consecutive frames per blink

# Device detection
ElectronicDeviceDetector.CONF_THRESH = 0.40   # YOLO confidence threshold
ElectronicDeviceDetector.SKIP_FRAMES = 5      # run YOLO every N frames

# Movement / fidgeting
HeadMovementTracker.fidget_thresh = 32   # variance threshold
```

To add prohibited items, register their COCO class IDs in `DEVICE_CLASS_IDS` and include them in either `CHEAT_DEVICE_IDS` or `WARN_DEVICE_IDS`.

---

## Output

```
project/
├── smart_proctor_pro.py
├── face_landmarker.task            # auto-downloaded
├── yolov4-tiny.cfg                 # auto-downloaded
├── yolov4-tiny.weights             # auto-downloaded
├── coco.names                      # auto-downloaded
├── alarm.wav                       # auto-generated
├── logs/
│   ├── events_YYYYMMDD_HHMMSS.csv  # timestamped event log
│   └── report_YYYYMMDD_HHMMSS.json # session risk report
└── screenshots/
    └── YYYYMMDD_HHMMSS_shot001.jpg
```

### Sample Report

```json
{
  "session_id": "20240427_143012",
  "duration": "12m 30s",
  "total_cheat_events": 7,
  "risk_level": "Medium",
  "stats": {
    "head_left": 3,
    "eye_gaze_right": 2,
    "device_phone": 1,
    "multiple_faces": 1
  },
  "events": []
}
```

### Risk Levels

| Violations per minute | Risk level |
|---|---|
| 0 | Clean |
| < 2 | Low |
| 2 – 5 | Medium |
| > 5 | High |

---

## FAQ

**Does it work offline?**
Yes. After the first run downloads the models, no internet connection is required.

**Which platforms are supported?**
Windows, macOS, and Linux. The alarm uses `afplay` on macOS and `aplay` on Linux, and the camera backend falls back gracefully.

**How do I use an external webcam?**
Change the index in `cv2.VideoCapture(0, ...)` from `0` to `1` or `2`.

**How accurate is device detection?**
YOLOv4-tiny offers good real-time performance on CPU. For higher accuracy at the cost of speed, substitute full YOLOv4 or YOLOv8.

---

## Limitations and Responsible Use

- Automated flags indicate *possible* misconduct, not proof. Results should be reviewed by a human before any academic action is taken.
- Accuracy varies with lighting, camera quality, camera angle, eyewear, and individual facial characteristics.
- Candidates must be informed that video is being analyzed and must consent before use. Handle any stored logs and screenshots in line with applicable privacy laws and your institution's data-retention policy.

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes, then submit a pull request with a clear description of what changed and why.

---

## License

Released under the [MIT License](LICENSE).

---

## Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe): face landmark model
- [AlexeyAB / Darknet](https://github.com/AlexeyAB/darknet): YOLOv4-tiny weights
- [OpenCV](https://opencv.org): computer vision backbone
