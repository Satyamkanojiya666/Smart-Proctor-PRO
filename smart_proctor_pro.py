"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SMART PROCTOR PRO               🚀  [+ DEVICE DETECT]   ║
║  MediaPipe 0.10+ Tasks API · 478 Landmarks · 3D Pose · Iris Tracking       ║
║  Electronic Device Detection: Phone · Laptop · Tablet · Earphones          ║
╚══════════════════════════════════════════════════════════════════════════════╝

  INSTALL:
      pip install opencv-python mediapipe numpy

  FIRST RUN:
      Downloads face_landmarker.task (~30 MB) automatically to script folder.
      Downloads YOLOv4-tiny weights (~24 MB) for device detection.

  BEHAVIOR RULES:
      ✅ ALLOWED  → Eye Up / Down / Center, small natural head movement
      ❌ CHEAT    → Eye Left/Right, Head Left/Right, Head Tilt Up,
                    Face Tilt Down, Extended Eyes Closed, No Face,
                    Multiple Faces, New person entering frame,
                    📱 Phone / 💻 Laptop / 📋 Clipboard detected
      ⚠  WARN     → Blink abnormal, Too close/far, Fidgeting

  PRIORITY ORDER:
      0. Electronic Device Detected 🚨 (HIGHEST)
      1. Multiple Faces        🚨   2. Face Not Detected  🚨
      3. Head Left/Right       ❌   4. Eye Gaze Left/Right ❌
      5. Head Tilt Up/Down     ❌   6. Blink/Distance/Fidget ⚠

  CONTROLS:  ESC=end   D=debug   R=reset   S=screenshot
"""

# ── Project metadata ──────────────────────────────────────────────────────────
__author__  = "Satyam Kanojiya"
__github__  = "https://github.com/Satyamkanojiya666"
__version__ = "2.0.0"
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import mediapipe as mp
import time, json, csv, os, math, struct, wave
import threading, collections, urllib.request, sys
from datetime import datetime
from mediapipe.tasks        import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe              import Image, ImageFormat
import argparse
from iot_bridge import IoTNode, Fusion, IntegrityScore, Dashboard
import extras, webapp, signal

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL AUTO-DOWNLOAD  (face_landmarker.task, ~30 MB)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

# ─────────────────────────────────────────────────────────────────────────────
#  YOLO MODEL PATHS  (YOLOv4-tiny — fastest for CPU real-time)
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CFG_URL     = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
YOLO_NAMES_URL   = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

_script_dir  = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG     = os.path.join(_script_dir, "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(_script_dir, "yolov4-tiny.weights")
YOLO_NAMES   = os.path.join(_script_dir, "coco.names")

# COCO class IDs for electronic/prohibited devices
DEVICE_CLASS_IDS = {
    67: ("Cell Phone",  (0, 30, 255),   "phone"),
    63: ("Laptop",      (0, 60, 220),   "laptop"),
    64: ("Mouse",       (0, 100, 200),  "mouse"),
    66: ("Keyboard",    (0, 120, 200),  "keyboard"),
    65: ("Remote",      (0, 80, 200),   "remote"),
    73: ("Book/Notes",  (0, 140, 180),  "book"),
    74: ("Smartwatch",  (0, 40, 230),   "smartwatch"),
    0:  ("Extra Person",(200, 0, 160),  "person"),
}
CHEAT_DEVICE_IDS = {67, 63, 65, 0}      # phone, laptop/tablet, remote, 2nd person  -> hard violation
WARN_DEVICE_IDS  = {64, 66, 73, 74}     # mouse, keyboard, book/notes, watch        -> soft warning


def _dl(url, path, label):
    print(f"⬇  Downloading {label} …")
    def _prog(blk, blk_sz, total):
        done = blk * blk_sz
        if total > 0:
            pct = min(done * 100 // total, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%  ", end="", flush=True)
    urllib.request.urlretrieve(url, path, _prog)
    print(f"\n✓ Saved: {path}")


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print(f"✓ Face model: {MODEL_PATH}")
        return True
    print("⬇  Downloading face_landmarker.task (~30 MB) — first run only …")
    try:
        _dl(MODEL_URL, MODEL_PATH, "face_landmarker.task (~30 MB)")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def ensure_yolo():
    missing = []
    if not (os.path.exists(YOLO_CFG)     and os.path.getsize(YOLO_CFG)     > 1000):
        missing.append(("cfg",     YOLO_CFG_URL,     YOLO_CFG,     "yolov4-tiny.cfg (~8 KB)"))
    if not (os.path.exists(YOLO_NAMES)   and os.path.getsize(YOLO_NAMES)   > 100):
        missing.append(("names",   YOLO_NAMES_URL,   YOLO_NAMES,   "coco.names (~1 KB)"))
    if not (os.path.exists(YOLO_WEIGHTS) and os.path.getsize(YOLO_WEIGHTS) > 1_000_000):
        missing.append(("weights", YOLO_WEIGHTS_URL, YOLO_WEIGHTS, "yolov4-tiny.weights (~24 MB)"))

    for _, url, path, label in missing:
        try:
            _dl(url, path, label)
        except Exception as e:
            print(f"✗ YOLO download failed ({label}): {e}")
            print("  Device detection disabled — proceeding without it.")
            return None
    return YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  ELECTRONIC DEVICE DETECTOR  (YOLOv4-tiny)
# ─────────────────────────────────────────────────────────────────────────────
class ElectronicDeviceDetector:
    CONF_THRESH = 0.40
    NMS_THRESH  = 0.40
    INPUT_SIZE  = (416, 416)
    SKIP_FRAMES = 5

    def __init__(self, cfg, weights, names):
        self.net   = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        ln         = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()
        if out_layers.ndim == 1:
            self.output_layers = [ln[i - 1] for i in out_layers]
        else:
            self.output_layers = [ln[i[0] - 1] for i in out_layers]
        with open(names) as f:
            self.class_names = [l.strip() for l in f.readlines()]
        self._cache       = []
        self._frame_count = 0
        print("✓ YOLOv4-tiny loaded (Electronic Device Detection)")

    fresh = False
    def detect(self, frame):
        self._frame_count += 1
        self.fresh = False
        if self._frame_count % self.SKIP_FRAMES != 0:
            return self._cache
        self.fresh = True

        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.INPUT_SIZE,
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes, confs, class_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid    = int(np.argmax(scores))
                conf   = float(scores[cid])
                if cid not in DEVICE_CLASS_IDS: continue
                if conf < self.CONF_THRESH:      continue
                cx, cy, bw, bh = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                x = int(cx - bw/2); y = int(cy - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confs.append(conf)
                class_ids.append(cid)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.CONF_THRESH, self.NMS_THRESH)
        results = []
        if len(idxs):
            for i in (idxs.flatten() if hasattr(idxs, 'flatten') else idxs):
                cid = class_ids[i]
                lbl, clr, key = DEVICE_CLASS_IDS[cid]
                results.append((cid, lbl, clr, confs[i], *boxes[i]))
        if sum(1 for r in results if r[0] == 0) < 2:        # one person = the student himself
            results = [r for r in results if r[0] != 0]
        self._cache = results
        return results

    def draw(self, frame, detections):
        for cid, lbl, clr, conf, x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), clr, 3)
            badge = f"⚠ {lbl.upper()} ({conf*100:.0f}%)"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, y-bh-14), (x+bw+8, y), clr, -1)
            cv2.putText(frame, badge, (x+4, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if int(time.time()*3) % 2 == 0:
                cv2.putText(frame, "DEVICE DETECTED",
                            (x, y+h+26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 30, 255), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK INDICES  (MediaPipe 478-point model)
# ─────────────────────────────────────────────────────────────────────────────
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
L_EYE_LEFT = 362; L_EYE_RIGHT = 263
R_EYE_LEFT = 133; R_EYE_RIGHT = 33

POSE_IDS = [1, 152, 263, 33, 287, 57]
MODEL_3D = np.array([
    [   0.0,    0.0,    0.0],
    [   0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  ALARM
# ─────────────────────────────────────────────────────────────────────────────
def _gen_wav(path, freq=880, dur=0.45, vol=0.85):
    sr, n = 44100, int(44100 * dur)
    with wave.open(path, "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        for i in range(n):
            env = 1.0 - (i / n) ** 2
            v   = int(32767 * vol * env * math.sin(2 * math.pi * freq * i / sr))
            f.writeframes(struct.pack('<h', v))

def _play():
    try:
        import winsound; winsound.Beep(880, 450)
    except ImportError:
        wav = "alarm.wav"
        if not os.path.exists(wav): _gen_wav(wav)
        cmd = (["afplay", wav] if sys.platform == "darwin"
               else ["aplay", "-q", wav])
        try:
            import subprocess
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception: pass

def alarm():
    threading.Thread(target=_play, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, sid):
        os.makedirs("logs", exist_ok=True)
        self.events = []
        self._f = open(f"logs/events_{sid}.csv", "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestamp", "event", "detail"])

    def log(self, event, detail=""):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._w.writerow([ts, event, detail])
        self._f.flush()
        self.events.append({"time": ts, "event": event, "detail": detail})
        print(f"  [{ts}] ⚑  {event}  {detail}")

    def close(self): self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_report(sid, cheat_count, dur, stats, events):
    rate = cheat_count / max(dur / 60, 1)
    risk = ("Clean ✓"   if rate == 0 else
            "Low ⚠"     if rate < 2  else
            "Medium ⚠⚠" if rate < 5  else "High 🚨")
    report = {
        "session_id":         sid,
        "student":            student,
        "generated_at":       datetime.now().isoformat(),
        "author":             __author__,
        "github":             __github__,
        "engine":             "MediaPipe 0.10+ FaceLandmarker + YOLOv4-tiny Device Detection",
        "duration":           f"{int(dur)//60}m {int(dur)%60}s",
        "total_cheat_events": cheat_count,
        "risk_level":         risk,
        "stats":              stats,
        "final_integrity_score": round(integrity.v),
        "integrity_grade":     integrity.grade,
        "events":             events,
    }
    path = f"logs/report_{sid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path, risk


# ─────────────────────────────────────────────────────────────────────────────
#  COOLDOWN
# ─────────────────────────────────────────────────────────────────────────────
class Cooldown:
    def __init__(self, seconds=2.0):
        self._cd = seconds; self._last = {}
    def ready(self, key):
        now = time.time()
        if now - self._last.get(key, 0) >= self._cd:
            self._last[key] = now; return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  3D HEAD POSE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
class HeadPoseEstimator:
    def __init__(self, W, H):
        f = float(W)
        self.cam  = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros((4, 1))
        self.W, self.H = W, H

    def _euler_from_R(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(-R[2,0], sy))
            yaw   = math.degrees(math.atan2( R[1,0], R[0,0]))
            roll  = math.degrees(math.atan2( R[2,1], R[2,2]))
        else:
            pitch = math.degrees(math.atan2(-R[2,0], sy))
            yaw   = math.degrees(math.atan2(-R[1,2], R[1,1]))
            roll  = 0.0
        return round(pitch, 1), round(yaw, 1), round(roll, 1)

    def from_matrix(self, mat4x4):
        R = np.array(mat4x4, dtype=np.float64)[:3, :3]
        return self._euler_from_R(R)

    def from_landmarks(self, lms):
        try:
            pts2d = np.array(
                [[lms[i].x*self.W, lms[i].y*self.H] for i in POSE_IDS],
                dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, self.cam, self.dist,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: return 0.0, 0.0, 0.0
            R, _ = cv2.Rodrigues(rvec)
            return self._euler_from_R(R)
        except Exception:
            return 0.0, 0.0, 0.0

    def draw_axes(self, frame, lms, length=70):
        try:
            pts2d = np.array(
                [[lms[i].x*self.W, lms[i].y*self.H] for i in POSE_IDS],
                dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, self.cam, self.dist,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: return
            axes3d = np.float32([[length,0,0],[0,length,0],[0,0,length],[0,0,0]])
            pts, _ = cv2.projectPoints(axes3d, rvec, tvec, self.cam, self.dist)
            pts    = pts.astype(int)
            orig   = tuple(pts[3].ravel())
            cv2.arrowedLine(frame, orig, tuple(pts[0].ravel()), (0,0,255),   2, tipLength=0.3)
            cv2.arrowedLine(frame, orig, tuple(pts[1].ravel()), (0,255,0),   2, tipLength=0.3)
            cv2.arrowedLine(frame, orig, tuple(pts[2].ravel()), (255,100,0), 2, tipLength=0.3)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  IRIS GAZE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
class IrisGazeEstimator:
    def _center(self, lms, ids, W, H):
        pts = np.array([[lms[i].x*W, lms[i].y*H] for i in ids])
        return pts.mean(axis=0)

    def estimate(self, lms, W, H, frame):
        try:
            li = self._center(lms, LEFT_IRIS,  W, H)
            ri = self._center(lms, RIGHT_IRIS, W, H)
            ll = np.array([lms[L_EYE_LEFT ].x*W, lms[L_EYE_LEFT ].y*H])
            lr = np.array([lms[L_EYE_RIGHT].x*W, lms[L_EYE_RIGHT].y*H])
            rl = np.array([lms[R_EYE_LEFT ].x*W, lms[R_EYE_LEFT ].y*H])
            rr = np.array([lms[R_EYE_RIGHT].x*W, lms[R_EYE_RIGHT].y*H])

            def hr(iris, el, er):
                span = np.linalg.norm(er - el)
                return float(np.dot(iris-el, er-el) / span**2) if span > 1 else 0.5

            def vr(iris, eye_ids):
                ys = [lms[i].y*H for i in eye_ids]
                top, bot = min(ys), max(ys)
                return (iris[1]-top)/(bot-top) if (bot-top) > 1 else 0.5

            avg_h = (hr(li, ll, lr) + hr(ri, rl, rr)) / 2.0
            avg_v = (vr(li, LEFT_EYE) + vr(ri, RIGHT_EYE)) / 2.0

            for pt in [li, ri]:
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 200, 255), 2)
                cv2.circle(frame, tuple(pt.astype(int)), 2, (0,  60, 255), -1)

            dbg = {"L_h": round(hr(li,ll,lr),2), "R_h": round(hr(ri,rl,rr),2),
                   "avg_h": round(avg_h,2), "avg_v": round(avg_v,2)}

            if   avg_v < 0.30: gaze = "Eye: Looking Up"
            elif avg_v > 0.75: gaze = "Eye: Looking Down"
            elif avg_h < 0.35: gaze = "Eye Gaze: Left ◀"
            elif avg_h > 0.65: gaze = "Eye Gaze: Right ▶"
            else:              gaze = "Eye: Center"
            return gaze, dbg
        except Exception:
            return "Eyes Not Detected", {}


# ─────────────────────────────────────────────────────────────────────────────
#  EAR BLINK DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class EARBlinkDetector:
    EAR_THRESH = 0.20
    CONSEC     = 3

    def __init__(self):
        self.history = collections.deque(maxlen=900)
        self._cnt    = 0
        self._state  = False
        self._cf     = 0

    def _ear(self, lms, ids, W, H):
        p = np.array([[lms[i].x*W, lms[i].y*H] for i in ids])
        A = np.linalg.norm(p[1]-p[5]); B = np.linalg.norm(p[2]-p[4])
        C = np.linalg.norm(p[0]-p[3])
        return (A+B) / (2.0*C + 1e-6)

    def update(self, lms, W, H):
        try:
            ear    = (self._ear(lms, LEFT_EYE, W, H) +
                      self._ear(lms, RIGHT_EYE, W, H)) / 2.0
            closed = ear < self.EAR_THRESH
        except Exception:
            ear, closed = 0.0, True

        if closed:
            self._cnt += 1; self._cf += 1
        else:
            if self._cnt >= self.CONSEC and self._state:
                self.history.append(time.time())
            self._cnt = 0; self._cf = 0
        self._state = closed

        now  = time.time()
        bpm  = len([t for t in self.history if now-t < 60])
        susp = (bpm < 6 or bpm > 40) if len(self.history) > 5 else False
        ext  = self._cf > 45
        return closed, round(ear, 3), bpm, susp, ext


# ─────────────────────────────────────────────────────────────────────────────
#  FACE DISTANCE CHECKER
# ─────────────────────────────────────────────────────────────────────────────
class FaceDistanceChecker:
    def check(self, face_w, frame_w):
        r = face_w / max(frame_w, 1)
        if r > 0.60: return "Too Close!", r
        if r < 0.10: return "Too Far!",   r
        return "OK", r


# ─────────────────────────────────────────────────────────────────────────────
#  HEAD MOVEMENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class HeadMovementTracker:
    def __init__(self, history=20, move_thresh=10, fidget_thresh=32):
        self.pos = collections.deque(maxlen=history)
        self.mt  = move_thresh
        self.ft  = fidget_thresh

    def update(self, cx, cy):
        self.pos.append((cx, cy))
        if len(self.pos) < 5: return "Stable", 0.0, False
        r  = list(self.pos)[-5:]
        dx = r[-1][0] - r[0][0]; dy = r[-1][1] - r[0][1]
        if abs(dx) > abs(dy):
            d = ("Moving ◀" if dx < -self.mt else "Moving ▶" if dx > self.mt else "Stable")
        else:
            d = ("Moving ▲" if dy < -self.mt else "Moving ▼" if dy > self.mt else "Stable")
        xs  = [p[0] for p in self.pos]; ys = [p[1] for p in self.pos]
        var = float(np.std(xs) + np.std(ys))
        return d, round(var, 1), var > self.ft


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-FACE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
FACE_COLORS = [(0,220,0),(0,60,255),(0,200,255),(255,0,200),(0,165,255)]

class MultiFaceTracker:
    def __init__(self):
        self.tracked = {}; self._next_id = 1

    def update(self, rects):
        if not len(rects): self.tracked = {}; return [], False
        new, used, new_entry = {}, set(), False
        for (x, y, w, h) in rects:
            cx, cy    = x+w//2, y+h//2
            best_id, best_d = None, float("inf")
            for fid, (px, py) in self.tracked.items():
                d = math.hypot(cx-px, cy-py)
                if d < best_d and fid not in used:
                    best_d, best_id = d, fid
            if best_id and best_d < 140:
                new[best_id] = (cx, cy); used.add(best_id)
            else:
                new[self._next_id] = (cx, cy); used.add(self._next_id)
                if self._next_id > 1: new_entry = True
                self._next_id += 1
        self.tracked = new
        return list(zip(rects, list(new.keys()))), new_entry


# ─────────────────────────────────────────────────────────────────────────────
#  STATUS CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
HARD_CHEAT = {
    "Face Not Detected!",
    "Head: Looking Left ◀", "Head: Looking Right ▶",
    "Head Tilt Up ▲",       "Face Tilt Down ▼",
    "Eye Gaze: Left ◀",     "Eye Gaze: Right ▶",
}
SOFT_WARN = {
    "Eyes Closed (Extended)", "Eyes Not Detected",
    "Too Close!", "Too Far!",
}

def classify(label):
    if "Multiple" in label: return "CHEAT"
    if "Device"   in label: return "CHEAT"
    if "Notes"    in label: return "WARN"
    if label in HARD_CHEAT:  return "CHEAT"
    if label in SOFT_WARN:   return "WARN"
    return "OK"


# ─────────────────────────────────────────────────────────────────────────────
#  UI RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def draw_ui(frame, status, sc, cheat_ct, elapsed,
            fps, bpm, ear, dist_ratio, variance, move_dir,
            gaze_label, face_count, pitch, yaw, roll,
            gaze_debug, show_debug, sid, device_detections):
    h, w = frame.shape[:2]; t = time.time()
    is_cheat = sc == "CHEAT"; is_warn = sc == "WARN"
    clr = ((30,30,255) if is_cheat else (30,160,255) if is_warn else (30,210,30))

    cv2.rectangle(frame, (0, 0), (w, 118), (10, 10, 10), -1)
    pr = 15 if (is_cheat and int(t*5)%2==0) else 10
    cv2.circle(frame, (28, 40), pr, clr, -1)
    cv2.putText(frame, f"Status: {status}",
                (52, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, clr, 2)
    mm, ss = elapsed//60, elapsed%60
    cv2.putText(frame, f"Cheats: {cheat_ct}",     (16, 90),      cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200,200,200), 1)
    cv2.putText(frame, f"Time: {mm:02d}:{ss:02d}", (w//2-55, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200,200,200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}",          (w-115, 90),   cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120,120,120), 1)
    gc = (30,30,255) if ("Left" in gaze_label or "Right" in gaze_label) else (140,220,255)
    cv2.putText(frame, gaze_label, (w-270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, gc, 1)
    bc = (30,30,255) if face_count > 1 else (30,180,30)
    cv2.rectangle(frame, (w-118, 100), (w-6, 128), bc, -1)
    cv2.putText(frame, f"Faces: {face_count}", (w-112, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

    cv2.rectangle(frame, (0, 118), (w, 142), (18, 18, 18), -1)
    pc = (30,30,255) if abs(pitch) > 20 else (160,200,160)
    yc = (30,30,255) if abs(yaw)   > 20 else (160,200,160)
    cv2.putText(frame, f"PITCH:{pitch:+.1f}°", (14,  136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, pc,          1)
    cv2.putText(frame, f"YAW:{yaw:+.1f}°",     (162, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, yc,          1)
    cv2.putText(frame, f"ROLL:{roll:+.1f}°",    (296, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,100), 1)
    cv2.putText(frame, f"EAR:{ear:.3f}",         (424, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,200), 1)

    dev_panel_y = 160
    cv2.rectangle(frame, (0, dev_panel_y), (190, dev_panel_y+18), (25, 10, 10), -1)
    cv2.putText(frame, "[ DEVICE SCAN ]", (6, dev_panel_y+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,60,60), 1)
    if device_detections:
        for i, (cid, lbl, clr_d, conf, *_) in enumerate(device_detections[:4]):
            dy = dev_panel_y + 30 + i*20
            badge_clr = (0,30,255) if cid in CHEAT_DEVICE_IDS else (30,140,255)
            cv2.putText(frame, f"  📱 {lbl} {conf*100:.0f}%",
                        (6, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_clr, 1)
    else:
        cv2.putText(frame, "  No devices", (6, dev_panel_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60,160,60), 1)

    if is_cheat:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 142), (w, h-44), (0, 0, 100), -1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
        if int(t*3) % 2 == 0:
            txt = "! CHEATING DETECTED !"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.15, 3)
            tx, ty = (w-tw)//2, h//2+24
            cv2.putText(frame, txt, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0,0,0),       4)
            cv2.putText(frame, txt, (tx,   ty),   cv2.FONT_HERSHEY_SIMPLEX, 1.15, (30,30,255),   3)
    elif is_warn and int(t*2) % 2 == 0:
        txt = "SUSPICIOUS BEHAVIOR"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, txt, ((w-tw)//2, h//2+24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,160,255), 2)

    mc = (255,200,80) if move_dir != "Stable" else (180,180,180)
    infos = [
        ("DIST",  f"{int(dist_ratio*100)}%", (255,90,90)  if (dist_ratio>0.6 or 0<dist_ratio<0.10) else (180,180,180)),
        ("STAB",  f"{variance:.0f}",          (255,170,60) if variance > 32 else (180,180,180)),
        ("BLINK", f"{bpm}/min",               (255,90,90)  if (bpm < 6 or bpm > 40) else (180,180,180)),
        ("MOVE",  move_dir, mc),
    ]
    py = dev_panel_y + 110
    for lbl, val, c in infos:
        cv2.putText(frame, f"{lbl}: {val}", (12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, c, 1)
        py += 22

    if show_debug and gaze_debug:
        dpx, dpy = w-215, h-180
        cv2.rectangle(frame, (dpx-6, dpy-22), (w-6, h-50), (16,16,16), -1)
        cv2.putText(frame, "[ Iris Debug ]", (dpx, dpy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,50), 1)
        for i, (k, v) in enumerate(gaze_debug.items()):
            cv2.putText(frame, f"  {k}: {v}", (dpx, dpy+18+i*16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

    cv2.rectangle(frame, (0, h-44), (w, h), (10,10,10), -1)
    cv2.putText(frame, f"Session: {sid}   ESC=end  D=debug  R=reset  S=screenshot",
                (12, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70,70,70), 1)
    cv2.putText(frame, f"{__github__}  |  Smart Proctor Pro + YOLOv4-tiny Device Detection",
                (w-470, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (50,90,50), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────────────────────
print(__doc__)
print("=" * 70)

if not ensure_model():
    input("\nCannot continue without face model. Press Enter to exit...")
    exit(1)

yolo_paths      = ensure_yolo()
device_detector = None
if yolo_paths:
    try:
        device_detector = ElectronicDeviceDetector(*yolo_paths)
    except Exception as e:
        print(f"✗ YOLO init failed: {e}  →  Device detection disabled.")

face_cas = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if not os.path.exists("alarm.wav"):
    _gen_wav("alarm.wav"); print("✓ alarm.wav generated")

cap = None
for backend, name in [(cv2.CAP_DSHOW,"DirectShow"),
                      (cv2.CAP_MSMF, "Media Foundation"),
                      (0,            "Default")]:
    try:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: print(f"✓ Camera: {name}"); break
            cap.release()
    except Exception: pass

if cap is None or not cap.isOpened():
    print("ERROR: Cannot open camera.")
    input("Press Enter to exit..."); exit(1)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
ret, _frame = cap.read()
H_CAM, W_CAM = _frame.shape[:2]

base_opts = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
fl_opts   = mp_vision.FaceLandmarkerOptions(
    base_options                          = base_opts,
    running_mode                          = mp_vision.RunningMode.VIDEO,
    num_faces                             = 4,
    min_face_detection_confidence         = 0.60,
    min_face_presence_confidence          = 0.55,
    min_tracking_confidence               = 0.50,
    output_face_blendshapes               = False,
    output_facial_transformation_matrixes = True,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(fl_opts)
print("✓ MediaPipe FaceLandmarker loaded\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION
# ─────────────────────────────────────────────────────────────────────────────
sid          = datetime.now().strftime("%Y%m%d_%H%M%S")
logger       = Logger(sid)
pose_est     = HeadPoseEstimator(W_CAM, H_CAM)
gaze_est     = IrisGazeEstimator()
blink_det    = EARBlinkDetector()
dist_chk     = FaceDistanceChecker()
move_tracker = HeadMovementTracker()
face_tracker = MultiFaceTracker()
cd           = Cooldown(2.0)
alarm_cd     = Cooldown(3.5)

# ── IoT layer ────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--port", default=None, help="ESP32 serial port (COM3, /dev/ttyUSB0); auto-detect if omitted")
_ap.add_argument("--simulate", action="store_true", help="run without hardware (keys T/A/P/B simulate sensors)")
_ap.add_argument("--dash-port", type=int, default=8000)
_ap.add_argument("--minutes", type=float, default=0, help="exam duration; auto-ends (0 = unlimited)")
_ap.add_argument("--no-voice", action="store_true", help="disable voice warnings")
_ap.add_argument("--no-login", action="store_true", help="(unused, login is on the web page now)")
_ap.add_argument("--show-window", action="store_true", help="show the OpenCV window (default: runs in background)")
_ap.add_argument("--py-voice", action="store_true", help="also speak warnings from Python (browser already speaks them)")
_ap.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
_args, _ = _ap.parse_known_args()
iot        = IoTNode(port=_args.port, simulate=_args.simulate)
fusion     = Fusion()
integrity  = IntegrityScore()
dashboard  = webapp.WebApp(_args.dash_port, open_browser=not _args.no_browser)
web        = dashboard
student    = {"name": "-", "roll": "-"}
max_viol   = 10
id_checks = id_ok = id_bad = 0; id_min = None; _idt = 0.0
voice      = extras.Voice(enabled=_args.py_voice and not _args.no_voice)
web.sim_cb = lambda k: (iot.sim_button() if k == "btn" else iot.sim_toggle(k)) if iot.mode == "simulator" else None
signal.signal(signal.SIGINT, lambda *a: setattr(web, "stop", True))
pattern    = extras.PatternWatch()
dev_track  = extras.DeviceTracker()
timeline   = extras.Timeline()
time_limit = 0
os.makedirs("evidence", exist_ok=True)
frame      = None
print(f"✓ IoT      : {iot.mode}")
print(f"✓ Dashboard: http://localhost:{dashboard.port}  (phone pe same WiFi IP se bhi khulega)" if dashboard.port else "✗ Dashboard port busy")

start        = time.time()
cheat_count  = 0
frame_count  = 0
show_debug   = False
fps_timer    = time.time()
fps          = 0.0
warn_buf     = collections.deque(maxlen=10)
screenshot_n = 0
frame_ts_ms  = 0

stats = {
    "head_left":0, "head_right":0, "head_tilt_up":0, "face_tilt_down":0,
    "eye_gaze_left":0, "eye_gaze_right":0, "no_face":0, "multiple_faces":0,
    "eyes_closed_ext":0, "too_close":0, "too_far":0,
    "blink_abnormal":0, "fidgeting":0,
    "device_phone":0, "device_laptop":0, "device_keyboard":0,
    "device_mouse":0, "device_remote":0, "device_book":0,
    "device_smartwatch":0, "device_person":0,
}

def smooth_status(s):
    warn_buf.append(s)
    return max(set(warn_buf), key=warn_buf.count)

def flag_hard(event, detail, stat_key=None):
    global cheat_count
    if not web.active: return
    if stat_key: stats[stat_key] = stats.get(stat_key, 0) + 1
    if cd.ready(event):
        cheat_count += 1
        logger.log(event, detail)
        web.notify(event, True)
        integrity.hit("hard")
        if frame is not None:
            cv2.imwrite(f"evidence/{sid}_{datetime.now().strftime('%H%M%S')}_{event}.jpg", frame)
        iot.send("BUZZ:400")
        pattern.add(event)
        voice.say_event(event)
        if alarm_cd.ready("alarm"): alarm()

def flag_soft(event, detail, stat_key=None):
    if not web.active: return
    if stat_key: stats[stat_key] = stats.get(stat_key, 0) + 1
    if cd.ready("s_"+event):
        logger.log(f"[WARN] {event}", detail)
        integrity.hit("soft")

print(f"✓ Session  : {sid}")
print(f"✓ Engine   : MediaPipe 0.10+ FaceLandmarker + YOLOv4-tiny")
print(f"✓ Devices  : {'ACTIVE (YOLOv4-tiny)' if device_detector else 'DISABLED'}")
print("Controls   : ESC=end  D=debug  R=reset  S=screenshot\n")


def finalize_session(reason):
    global time_limit
    total = time.time() - start
    logger.close()
    _ph = web.fid.photos(student["roll"])
    _idinfo = {"checks": id_checks, "ok": id_ok, "min": id_min, "photo": _ph[0] if _ph else None,
               "pre_face": bool(web.pre.get("face")), "pre_voice": web.pre.get("voice") or "-"}
    rp, risk = save_report(sid, cheat_count, total, stats, logger.events)
    hp = extras.make_html_report(sid, student, logger.events, stats, round(integrity.v), integrity.grade,
                                 total, cheat_count, timeline.pts, iot.mode, open_browser=False,
                                 identity=_idinfo)
    web.finish(reason, sid, cheat_count, round(integrity.v), integrity.grade, risk, rp, hp, total, _idinfo)
    time_limit = 0
    iot.send("STATE:OK")
    print(f"  [SESSION END] {student['name']} ({student['roll']}) - {reason} - integrity {round(integrity.v)} - {hp}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    clean = frame.copy(); web.raw = clean      # untouched frame for face-ID (no overlays drawn)

    # ── exam session control (driven from the web page) ──
    if web.stop: break
    if web.pending_start:
        _u = web.pending_start; web.pending_start = None
        sid = datetime.now().strftime("%Y%m%d_%H%M%S"); logger.close(); logger = Logger(sid)
        integrity = IntegrityScore(); cheat_count = 0; stats = {k: 0 for k in stats}
        pattern = extras.PatternWatch(); dev_track = extras.DeviceTracker(); timeline = extras.Timeline(); fusion = Fusion(); cd = Cooldown(2.0)
        student = {"name": _u["name"], "roll": _u["roll"]}
        time_limit = int(web.exam["minutes"] * 60); max_viol = web.max_v
        start = time.time(); warn_buf.clear()
        id_checks = id_ok = id_bad = 0; id_min = None; _idt = time.time() + 8
        web.begin(sid, _u)
        print(f"  [SESSION START] {student['name']} ({student['roll']})  {sid}")
    if web.active:
        _why = ("submitted by student" if web.pending_end == "submitted" else
                web.pending_end if web.pending_end else
                "time over" if time_limit and time.time() - start >= time_limit else
                "too many violations" if cheat_count >= max_viol else None)
        if _why: finalize_session(_why)
    for _lv, _ev, _dt in web.drain():
        (flag_hard if _lv == "hard" else flag_soft)(_ev, _dt, None)
        if web.active: stats[_ev] = stats.get(_ev, 0) + 1
    frame_count  += 1
    frame_ts_ms  += 33

    if frame_count % 10 == 0:
        fps = 10 / max(time.time()-fps_timer, 0.001)
        fps_timer = time.time()

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_img, frame_ts_ms)

    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = list(face_cas.detectMultiScale(gray, 1.25, 5, minSize=(60, 60)))
    labeled_f, new_entry = face_tracker.update(haar_faces)

    raw_status   = "Normal"
    status_class = "OK"
    gaze_label   = "—"
    gaze_debug   = {}
    dist_ratio   = 0.0
    variance     = 0.0
    move_dir     = "Stable"
    blink_bpm    = 0
    ear_val      = 0.0
    pitch = yaw  = roll = 0.0

    mp_count   = len(result.face_landmarks) if result.face_landmarks else 0
    face_count = max(mp_count, len(haar_faces))
    elapsed    = int(time.time() - start)

    # ════════════════════════════════════════════════════════════
    # PRIORITY 0: ELECTRONIC DEVICE DETECTION
    # ════════════════════════════════════════════════════════════
    device_detections = []
    if device_detector:
        device_detections = device_detector.detect(frame)
        device_detector.draw(frame, device_detections)
        if device_detector.fresh:
            dev_track.update({d[0] for d in device_detections})
        for cid, lbl, clr_d, conf, *_ in device_detections:
            if not dev_track.confirmed(cid): continue
            ename = "device_" + lbl.lower().replace(" ", "_").replace("/", "_")
            key = f"device_{DEVICE_CLASS_IDS[cid][2]}"
            if cid in CHEAT_DEVICE_IDS:
                detail = f"{lbl} detected (conf={conf*100:.0f}%)"
                if raw_status == "Normal":
                    raw_status   = f"Device: {lbl} Detected! 📵"
                    status_class = "CHEAT"
                if dev_track.ready(cid): flag_hard(ename, detail, key)
            elif cid in WARN_DEVICE_IDS:
                detail = f"{lbl} in frame (conf={conf*100:.0f}%)"
                if status_class == "OK":
                    raw_status   = f"Suspicious: {lbl} Visible ⚠"
                    status_class = "WARN"
                if dev_track.ready(cid): flag_soft(ename, detail, key)

    # ════════════════════════════════════════════════════════════
    # PRIORITY 1: MULTIPLE FACES
    # ════════════════════════════════════════════════════════════
    if face_count > 1:
        for (fx, fy, fw, fh), fid in labeled_f:
            clr = FACE_COLORS[(fid-1) % len(FACE_COLORS)]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), clr, 3)
            badge = f"FACE #{fid}"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            cv2.rectangle(frame, (fx, fy-bh-10), (fx+bw+8, fy), clr, -1)
            cv2.putText(frame, badge, (fx+4, fy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
            if fid > 1:
                cv2.putText(frame, "INTRUDER", (fx, fy+fh+22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,30,255), 2)
        detail = f"{face_count} faces" + (" — NEW PERSON ENTERED" if new_entry else "")
        if status_class != "CHEAT":
            raw_status   = f"Multiple Faces! ({face_count})"
            status_class = "CHEAT"
        flag_hard("multiple_faces", detail, "multiple_faces")
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            _, ear_val, blink_bpm, _, _ = blink_det.update(lms, W, H)
            move_dir, variance, _ = move_tracker.update(int(lms[1].x*W), int(lms[1].y*H))

    # ════════════════════════════════════════════════════════════
    # PRIORITY 2: NO FACE
    # ════════════════════════════════════════════════════════════
    elif face_count == 0 or not result.face_landmarks:
        if status_class != "CHEAT":
            raw_status   = "Face Not Detected!"
            status_class = "CHEAT"
        flag_hard("no_face", "No face detected", "no_face")
        move_tracker.update(W//2, H//2)

    # ════════════════════════════════════════════════════════════
    # SINGLE FACE — FULL ANALYSIS
    # ════════════════════════════════════════════════════════════
    else:
        lms = result.face_landmarks[0]
        step = max(1, len(lms)//80)
        for i in range(0, len(lms), step):
            cv2.circle(frame, (int(lms[i].x*W), int(lms[i].y*H)), 1, (30,80,30), -1)

        xs = [int(l.x*W) for l in lms]; ys = [int(l.y*H) for l in lms]
        fx, fy = min(xs), min(ys); fw, fh = max(xs)-fx, max(ys)-fy
        fcx, fcy = fx+fw//2, fy+fh//2
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,200,0), 2)

        move_dir, variance, fidgeting = move_tracker.update(fcx, fcy)
        if fidgeting: flag_soft("fidgeting", f"var={variance:.1f}", "fidgeting")

        dist_status, dist_ratio = dist_chk.check(fw, W)
        if dist_status != "OK":
            sk = "too_close" if "Close" in dist_status else "too_far"
            flag_soft(dist_status, f"ratio={dist_ratio:.2f}", sk)
            if status_class == "OK":
                raw_status = dist_status; status_class = "WARN"

        _, ear_val, blink_bpm, blink_sus, ext_closed = blink_det.update(lms, W, H)
        if blink_sus: flag_soft("blink_abnormal", f"bpm={blink_bpm}", "blink_abnormal")
        if ext_closed:
            flag_soft("eyes_closed_ext", "shut >1.5s", "eyes_closed_ext")
            if status_class == "OK":
                raw_status = "Eyes Closed (Extended)"; status_class = "WARN"

        if (result.facial_transformation_matrixes and
                len(result.facial_transformation_matrixes) > 0):
            mat_data = result.facial_transformation_matrixes[0].data
            pitch, yaw, roll = pose_est.from_matrix(mat_data)
        else:
            pitch, yaw, roll = pose_est.from_landmarks(lms)

        pose_est.draw_axes(frame, lms, length=70)

        if yaw < -20:
            if status_class != "CHEAT": raw_status = "Head: Looking Left ◀"; status_class = "CHEAT"
            flag_hard("head_left", f"yaw={yaw:.1f}°", "head_left")
        elif yaw > 20:
            if status_class != "CHEAT": raw_status = "Head: Looking Right ▶"; status_class = "CHEAT"
            flag_hard("head_right", f"yaw={yaw:.1f}°", "head_right")

        if pitch > 20:
            if status_class != "CHEAT": raw_status = "Head Tilt Up ▲"; status_class = "CHEAT"
            flag_hard("head_tilt_up", f"pitch={pitch:.1f}°", "head_tilt_up")
        elif pitch < -20:
            if status_class != "CHEAT": raw_status = "Face Tilt Down ▼"; status_class = "CHEAT"
            flag_hard("face_tilt_down", f"pitch={pitch:.1f}°", "face_tilt_down")

        gaze_label, gaze_debug = gaze_est.estimate(lms, W, H, frame)

        if gaze_label == "Eye Gaze: Left ◀":
            stats["eye_gaze_left"] += 1
            if status_class != "CHEAT": raw_status = gaze_label; status_class = "CHEAT"
            flag_hard("eye_gaze_left", "Iris shifted LEFT")
        elif gaze_label == "Eye Gaze: Right ▶":
            stats["eye_gaze_right"] += 1
            if status_class != "CHEAT": raw_status = gaze_label; status_class = "CHEAT"
            flag_hard("eye_gaze_right", "Iris shifted RIGHT")
        elif gaze_label == "Eyes Not Detected":
            flag_soft("eyes_not_detected", "landmarks missing")
            if status_class == "OK": raw_status = gaze_label; status_class = "WARN"

    # ── IoT sensor fusion ──
    sensors = iot.read()
    for lvl, ev, det in fusion.check(sensors, face_count):
        (flag_hard if lvl == "hard" else flag_soft)(ev, det, None)
        stats[ev] = stats.get(ev, 0) + 1
        if lvl == "hard" and status_class != "CHEAT":
            raw_status, status_class = f"IoT: {ev.replace('_',' ').title()}", "CHEAT"

    # ── continuous identity check (live face vs enrolled photo) ──
    if (web.active and web.exam.get("require_identity") and web.fid.ready and web.fid.enrolled(student["roll"])
            and time.time() - _idt >= 5 and face_count == 1 and abs(yaw) < 25 and abs(pitch) < 25):
        _idt = time.time()
        _sc = web.fid.check(clean, student["roll"])
        if _sc is not None:
            id_checks += 1; id_min = round(_sc if id_min is None else min(id_min, _sc), 3)
            if _sc >= web.id_thr * 0.85: id_ok += 1; id_bad = 0
            else:
                id_bad += 1
                if id_bad >= 3:
                    flag_hard("identity_mismatch", f"live face does not match enrolled photo (similarity {_sc:.2f})", None)
                    stats["identity_mismatch"] = stats.get("identity_mismatch", 0) + 1; id_bad = 0
            dashboard.update(idscore=round(_sc, 2))

    smoothed     = smooth_status(raw_status)
    smooth_class = classify(smoothed)
    if "IoT:" in smoothed: smooth_class = "CHEAT"
    iot.send("STATE:" + smooth_class)
    score = integrity.tick()
    for ev, det in pattern.check():
        flag_hard(ev, det, None)
        stats[ev] = stats.get(ev, 0) + 1
    timeline.add(time.time() - start, score)
    left_txt = extras.remaining_text(time_limit, time.time() - start)
    dashboard.update(status=smoothed, cls=smooth_class, score=round(score), grade=integrity.grade,
                     sensors=sensors, faces=face_count, cheats=cheat_count, mode=iot.mode,
                     history=list(integrity.history)[-120:], events=logger.events[-30:],
                     student=f"{student['name']} ({student['roll']})", left=left_txt)

    frame = draw_ui(
        frame, smoothed, smooth_class, cheat_count, elapsed,
        fps, blink_bpm, ear_val, dist_ratio, variance, move_dir,
        gaze_label, face_count, pitch, yaw, roll,
        gaze_debug, show_debug, sid, device_detections
    )

    cv2.putText(frame, f"IoT {sensors['dist']}cm snd:{sensors['sound']} pir:{sensors['pir']} | Integrity {score:.0f} ({integrity.grade})",
                (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.putText(frame, f"{student['name']} | {student['roll']}   {left_txt}", (10, H-36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    web.set_frame(frame)
    if _args.show_window:
        cv2.imshow("Smart Proctor Pro — Device Detection", frame)

    k = cv2.waitKey(1) & 0xFF
    if   k == 27: break
    elif k in (ord('d'), ord('D')): show_debug = not show_debug
    elif k in (ord('r'), ord('R')):
        cheat_count = 0; warn_buf.clear()
        print("  [RESET] Cheat counter cleared.")
    elif k in (ord('t'), ord('T')) and iot.mode == "simulator": print("  [SIM] talk:", iot.sim_toggle("talk"))
    elif k in (ord('a'), ord('A')) and iot.mode == "simulator": print("  [SIM] away:", iot.sim_toggle("away"))
    elif k in (ord('p'), ord('P')) and iot.mode == "simulator": print("  [SIM] pir :", iot.sim_toggle("pir"))
    elif k in (ord('b'), ord('B')) and iot.mode == "simulator": iot.sim_button()
    elif k in (ord('s'), ord('S')):
        os.makedirs("screenshots", exist_ok=True)
        screenshot_n += 1
        sp = f"screenshots/{sid}_shot{screenshot_n:03d}.jpg"
        cv2.imwrite(sp, frame)
        print(f"  [SCREENSHOT] {sp}")


# ─────────────────────────────────────────────────────────────────────────────
#  SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
if web.active: finalize_session("system stopped")
cap.release()
face_landmarker.close()
cv2.destroyAllWindows()
iot.close(); web.close()
logger.close()
print("\n  Smart Proctor stopped. Reports are in logs/ and data/results.json")
