"""
faceid.py - Face identity engine (OpenCV YuNet detector + SFace recogniser, ONNX).
Models (~37 MB) auto-download once into data/models/. Everything runs offline after that.
Enrolled photos: data/faces/<roll>_<n>.jpg   (max 3 per student)
"""
import os, re, glob, threading, urllib.request
import numpy as np, cv2

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "det": ("face_detection_yunet_2023mar.onnx", BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "rec": ("face_recognition_sface_2021dec.onnx", BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}
MAX_PHOTOS = 3


class FaceID:
    def __init__(self, faces_dir="data/faces", models_dir="data/models"):
        self.fd, self.md = faces_dir, models_dir
        os.makedirs(self.fd, exist_ok=True); os.makedirs(self.md, exist_ok=True)
        self.lock = threading.RLock()
        self.ready, self.err = False, "Face engine loading..."
        self.feats = {}
        threading.Thread(target=self._init, daemon=True).start()

    def _init(self):
        try:
            paths = {}
            for k, (fn, url) in MODELS.items():
                p = os.path.join(self.md, fn); paths[k] = p
                if not os.path.exists(p) or os.path.getsize(p) < 100000:
                    self.err = f"Downloading face model {fn} (one time, needs internet)..."
                    print("  [FaceID]", self.err)
                    urllib.request.urlretrieve(url, p + ".tmp"); os.replace(p + ".tmp", p)
            self.det = cv2.FaceDetectorYN.create(paths["det"], "", (320, 320), 0.85, 0.3, 5000)
            self.rec = cv2.FaceRecognizerSF.create(paths["rec"], "")
            self.ready, self.err = True, ""
            for f in {re.sub(r"_\d+\.jpg$", "", os.path.basename(x)) for x in glob.glob(self.fd + "/*.jpg")}:
                self._rebuild(f)
            print(f"  [FaceID] ready - {len(self.feats)} student(s) enrolled")
        except Exception as e:
            self.err = f"Face engine unavailable: {e}"
            print("  [FaceID]", self.err)

    # ---- core --------------------------------------------------------------
    def _faces(self, img):
        h, w = img.shape[:2]
        with self.lock:
            self.det.setInputSize((w, h))
            _, f = self.det.detect(img)
        return [] if f is None else list(f)

    def embed(self, img):
        """-> (feature or None, number_of_faces, face_width_px)"""
        if not self.ready or img is None: return None, 0, 0
        if max(img.shape[:2]) > 1000:
            s = 1000 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
        fs = self._faces(img)
        if not fs: return None, 0, 0
        best = max(fs, key=lambda r: r[2] * r[3])
        with self.lock:
            feat = self.rec.feature(self.rec.alignCrop(img, best)).copy()
        return feat, len(fs), int(best[2])

    def _cos(self, a, b):
        with self.lock:
            return float(self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    # ---- enrolment ---------------------------------------------------------
    def photos(self, roll):
        return sorted(glob.glob(os.path.join(self.fd, f"{roll}_*.jpg")))

    def enrolled(self, roll):
        return bool(self.feats.get(roll))

    def _rebuild(self, roll):
        out = []
        for p in self.photos(roll):
            f, n, _ = self.embed(cv2.imread(p))
            if f is not None: out.append(f)
        if out: self.feats[roll] = out
        else: self.feats.pop(roll, None)

    def add_photo(self, roll, img):
        if not self.ready: return False, self.err
        if len(self.photos(roll)) >= MAX_PHOTOS: return False, f"Max {MAX_PHOTOS} photos. Delete one first."
        f, n, w = self.embed(img)
        if n == 0: return False, "No face found in this photo. Use a clear, front-facing photo."
        if n > 1: return False, "More than one face found. Use a photo of only this student."
        if w < 60: return False, "Face is too small. Use a closer photo."
        used = {int(re.search(r"_(\d+)\.jpg$", p).group(1)) for p in self.photos(roll)}
        k = next(i for i in range(1, MAX_PHOTOS + 1) if i not in used)
        if max(img.shape[:2]) > 1000:
            s = 1000 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
        cv2.imwrite(os.path.join(self.fd, f"{roll}_{k}.jpg"), img)
        self._rebuild(roll)
        return True, f"Photo saved ({len(self.photos(roll))}/{MAX_PHOTOS})"

    def del_photo(self, roll, n):
        p = os.path.join(self.fd, f"{roll}_{int(n)}.jpg")
        if os.path.exists(p): os.remove(p)
        self._rebuild(roll)

    # ---- verification ------------------------------------------------------
    def check(self, img, roll):
        """cosine similarity of the largest face vs enrolled photos (None if no face)."""
        f, n, _ = self.embed(img)
        if f is None or not self.feats.get(roll): return None
        return max(self._cos(f, e) for e in self.feats[roll])
