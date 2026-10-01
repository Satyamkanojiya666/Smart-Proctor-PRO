"""Run this BEFORE the demo:  python check_setup.py   (or double-click check.bat)"""
import os, sys, socket
ok_all = True
def res(ok, name, fix=""):
    global ok_all
    ok_all &= ok
    print(("  [ OK ] " if ok else "  [FAIL] ") + name + ("" if ok else f"   ->  {fix}"))

print("\nSmart Proctor PRO - setup check\n")
res(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "install Python 3.10+")
try:
    import cv2; v = cv2.__version__
    res(int(v.split(".")[0]) < 5, f"OpenCV {v}", 'pip install "opencv-contrib-python<5"')
    res(hasattr(cv2, "FaceRecognizerSF"), "OpenCV face-recognition support", 'pip install "opencv-contrib-python<5"')
except Exception as e:
    res(False, "OpenCV", "pip install -r requirements.txt"); cv2 = None
for mod in ("mediapipe", "numpy", "serial"):
    try: __import__(mod); res(True, mod)
    except Exception: res(False, mod, "pip install -r requirements.txt")
try: import pyttsx3; res(True, "pyttsx3 (optional voice)")
except Exception: print("  [ -- ] pyttsx3 not installed (optional, browser speaks warnings anyway)")

for f, fix in (("exam_proctor_elite.py", ""), ("webapp.py", ""), ("faceid.py", ""), ("extras.py", ""), ("iot_bridge.py", ""),
               ("face_landmarker.task", "copy from the old folder"), ("yolov4-tiny.cfg", "copy from the old folder"),
               ("coco.names", "copy from the old folder")):
    res(os.path.exists(f), f, fix or "file missing - re-extract the zip")
w = "yolov4-tiny.weights"
if os.path.exists(w) and os.path.getsize(w) > 20_000_000: res(True, "yolov4-tiny.weights")
else: print("  [ -- ] yolov4-tiny.weights missing -> will auto-download on first run (internet on) or copy from old folder")
m = "data/models"
if all(os.path.exists(f"{m}/{x}") for x in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")):
    res(True, "face-ID models")
else: print("  [ -- ] face-ID models will download on first run (internet on, ~37 MB)")

s = socket.socket()
try: s.bind(("0.0.0.0", 8000)); res(True, "port 8000 free")
except OSError: res(False, "port 8000 free", "close the other run (Ctrl+C) or use --dash-port 8080")
finally: s.close()

if cv2:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    got = cap.isOpened() and cap.read()[0]; cap.release()
    res(bool(got), "webcam", "close Zoom/Camera app/other Python run, then retry")

print("\n" + ("ALL GOOD - run.bat chalao!" if ok_all else "Upar wale [FAIL] theek karo, phir dobara check_setup.py chalao.") + "\n")
