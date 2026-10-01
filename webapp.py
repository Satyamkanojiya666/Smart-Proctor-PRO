"""
webapp.py - Web front-end for Smart Proctor PRO
  * Register / Login (students) + admin (teacher) dashboard
  * Online exam portal (MCQ, timer, palette, review, fullscreen, tab-switch detection)
  * Live proctor video (MJPEG), live warnings, integrity score, sensors, simulator buttons
  * Results, CSV export, HTML evidence reports, exam editor
Pure Python standard library (+ cv2 for JPEG). Default admin: admin / admin123
"""
import json, os, re, time, queue, hashlib, secrets, threading, collections, webbrowser, csv, io, random, copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import numpy as np
import cv2
import extras, faceid

DATA = "data"
def _S(q, opts, ans, sec, marks=1, code=None, t="single"):
    d = {"type": t, "q": q, "options": opts, "answer": ans, "section": sec, "marks": marks}
    if code: d["code"] = code
    return d
def _M(q, opts, ans, sec, marks=2): return _S(q, opts, ans, sec, marks, t="multi")
def _T(q, ans, sec): return _S(q, ["True", "False"], 0 if ans else 1, sec, t="tf")
def _N(q, ans, sec, tol=0, marks=1): return {"type": "num", "q": q, "answer": ans, "tol": tol, "section": sec, "marks": marks}
def _X(q, acc, sec, marks=1): return {"type": "text", "q": q, "answer": acc, "section": sec, "marks": marks}

_A, _B, _C = "Programming & CS", "IoT & Electronics", "AI & Aptitude"
FULL_EXAM = {
    "title": "Full Demo Paper - Tech Fundamentals", "minutes": 20, "max_violations": 10, "attempts": 5,
    "negative": 0.25, "shuffle": True, "access_code": "",
    "require_identity": True, "voice_check": True, "id_threshold": 0.363,
    "questions": [
        _S("What is the output of this Python code?", ["64", "512", "256", "128"], 1, _A, code="print(2 ** 3 ** 2)"),
        _S("Time complexity of binary search on a sorted array is:", ["O(n)", "O(log n)", "O(n log n)", "O(1)"], 1, _A),
        _M("Which of these are MUTABLE in Python? (select all)", ["list", "tuple", "dict", "str"], [0, 2], _A),
        _T("A stack follows FIFO (first in, first out) order.", False, _A),
        _N("How many 1-bits are there in the binary form of decimal 13?", 3, _A),
        _X("Which HTTP method is normally used to submit a login form? (one word)", ["post"], _A),
        _S("What does this code print?", ["3", "4", "5", "Error"], 1, _A, code="x = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))"),
        _S("Which data structure is used in Breadth-First Search (BFS)?", ["Stack", "Queue", "Heap", "Tree"], 1, _A),
        _S("The logic voltage level of an ESP32 GPIO pin is:", ["5 V", "3.3 V", "12 V", "1.8 V"], 1, _B),
        _S("The HC-SR04 sensor measures distance using:", ["Infrared light", "Ultrasonic waves", "Laser", "Magnetic field"], 1, _B),
        _N("HC-SR04 echo pulse is 1160 microseconds (speed of sound 343 m/s). Distance in cm? (round to nearest whole number)", 20, _B, tol=1, marks=2),
        _S("A PIR sensor detects changes in:", ["Sound", "Infrared radiation", "Gas concentration", "Light intensity"], 1, _B),
        _T("A 220 ohm resistor is commonly placed in series with an LED to limit current.", True, _B),
        _M("Which are serial communication protocols used with microcontrollers? (select all)", ["I2C", "SPI", "UART", "JPEG"], [0, 1, 2], _B),
        _X("Name the lightweight publish/subscribe protocol widely used in IoT (4 letters).", ["mqtt"], _B),
        _S("According to Ohm's law:", ["V = I / R", "V = I x R", "V = I + R", "V = R / I"], 1, _B),
        _S("Which of these is a SUPERVISED learning task?", ["Clustering", "Classification", "PCA", "Association rules"], 1, _C),
        _S("In a CNN, which layer reduces the spatial size of feature maps?", ["Pooling", "Dense", "Dropout", "Softmax"], 0, _C),
        _T("Overfitting means a model does well on training data but poorly on unseen data.", True, _C),
        _N("A student scores 18 out of 24. What is the percentage?", 75, _C),
        _S("Find the next number: 2, 6, 12, 20, 30, ?", ["40", "42", "44", "36"], 1, _C),
        _N("A 120 m long train crosses a pole in 6 seconds. Its speed in m/s is?", 20, _C),
        _S("Odd one out:", ["Python", "C++", "Java", "MySQL"], 3, _C),
        _S("Face-recognition systems compare:", ["Fingerprints", "Face embeddings (numeric vectors)", "Passwords", "IP addresses"], 1, _C),
    ]}
QUICK_EXAM = {
    "title": "Quick Demo Paper (3 minutes)", "minutes": 3, "max_violations": 10, "attempts": 10,
    "negative": 0, "shuffle": False, "access_code": "",
    "require_identity": True, "voice_check": True, "id_threshold": 0.363,
    "questions": [
        _S("IoT stands for:", ["Internet of Things", "Input of Tech", "Internal Office Tools", "Integrated Online Test"], 0, "Basics"),
        _S("Which sensor detects motion?", ["PIR", "LDR", "DHT11", "MQ-2"], 0, "Basics"),
        _T("ESP32 has built-in Wi-Fi and Bluetooth.", True, "Basics"),
        _N("What is 15 + 27?", 42, "Basics"),
        _M("Which are INPUT devices? (select all)", ["Keyboard", "Mouse", "Printer", "Microphone"], [0, 1, 3], "Basics"),
        _X("Capital city of India? (one or two words)", ["new delhi", "delhi"], "Basics"),
    ]}


def normalize_exam(e):
    for k, v in (("minutes", 10), ("max_violations", 10), ("attempts", 3), ("require_identity", True),
                 ("voice_check", True), ("id_threshold", 0.363), ("negative", 0.0), ("shuffle", False), ("access_code", "")):
        e.setdefault(k, v)
    for q in e["questions"]:
        q.setdefault("type", "single"); q.setdefault("marks", 1); q.setdefault("section", "General")
    return e

def validate_exam(e):
    assert str(e["title"]).strip() and float(e["minutes"]) > 0
    e["minutes"], e["max_violations"], e["attempts"] = float(e["minutes"]), int(e.get("max_violations", 10)), int(e.get("attempts", 3))
    e["require_identity"], e["voice_check"] = bool(e.get("require_identity", True)), bool(e.get("voice_check", True))
    e["id_threshold"], e["negative"] = float(e.get("id_threshold", 0.363)), float(e.get("negative", 0))
    e["shuffle"], e["access_code"] = bool(e.get("shuffle", False)), str(e.get("access_code", "")).strip()
    assert e["questions"]
    for q in e["questions"]:
        t = q.get("type", "single"); assert t in ("single", "multi", "tf", "num", "text") and str(q["q"]).strip()
        q["type"], q["marks"] = t, float(q.get("marks", 1)); assert q["marks"] > 0
        q["section"] = str(q.get("section", "General")).strip() or "General"
        if t in ("single", "multi", "tf"):
            n = len(q["options"]); assert n >= 2
            if t == "multi":
                q["answer"] = sorted({int(a) for a in q["answer"]}); assert q["answer"] and all(0 <= a < n for a in q["answer"])
            else:
                q["answer"] = int(q["answer"]); assert 0 <= q["answer"] < n
        elif t == "num":
            q["answer"], q["tol"] = float(q["answer"]), float(q.get("tol", 0))
        else:
            a = q["answer"]; q["answer"] = [str(x).strip().lower() for x in (a if isinstance(a, list) else [a])]; assert q["answer"]
    return e


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 60000).hex()

def _jload(name, default):
    p = os.path.join(DATA, name)
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def _jsave(name, obj):
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class WebApp:
    def __init__(self, port=8000, open_browser=True):
        os.makedirs(DATA, exist_ok=True); os.makedirs("logs", exist_ok=True)
        self.users = _jload("users.json", {})
        self.results = _jload("results.json", [])
        self.lib = _jload("papers.json", None)
        if not self.lib:
            self.lib = {"active": "full", "papers": {"full": copy.deepcopy(FULL_EXAM), "quick": copy.deepcopy(QUICK_EXAM)}}
            old = _jload("exam.json", None)
            if old:
                try: self.lib["papers"]["classic"] = validate_exam(old)
                except Exception: pass
        for pp in self.lib["papers"].values(): normalize_exam(pp)
        if self.lib["active"] not in self.lib["papers"]: self.lib["active"] = next(iter(self.lib["papers"]))
        self.order, self.optmap, self.pub = [], [], {}
        self._activate(self.lib["active"])
        self.fid = faceid.FaceID()
        self.raw, self.verified, self.codes, self.pre = None, {}, {}, {}
        if "admin" not in self.users:
            s = secrets.token_hex(8)
            self.users["admin"] = {"name": "Teacher (Admin)", "roll": "admin", "salt": s,
                                   "hash": _hash("admin123", s), "role": "admin"}
            _jsave("users.json", self.users)
        self.sessions, self.lock = {}, threading.Lock()
        self.live = {"status": "-", "cls": "OK", "score": 100, "grade": "Trusted",
                     "sensors": {"dist": 0, "sound": 0, "pir": 0}, "faces": 0, "cheats": 0,
                     "mode": "-", "history": [], "events": []}
        self.active, self.pending_start, self.pending_end, self.stop = False, None, None, False
        self.sid, self.user, self.t0 = None, None, 0.0
        self.answers, self.marks = {}, {}
        self.notes, self.nid = collections.deque(maxlen=80), 0
        self.msg = {"id": 0, "text": ""}
        self.inq = queue.Queue()
        self.jpg, self._fn = b"", 0
        self.sim_cb, self.last_result = None, {}
        self.max_v = int(self.exam.get("max_violations", 10))
        app = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def _roll(self):
                for p in self.headers.get("Cookie", "").split(";"):
                    k, _, v = p.strip().partition("=")
                    if k == "sp_token": return app.sessions.get(v)
            def _user(self):
                r = self._roll(); return app.users.get(r) if r else None
            def _send(self, body, ct="application/json", code=200, cookie=None):
                if not isinstance(body, (bytes, bytearray)):
                    body = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ct + ("; charset=utf-8" if "text" in ct or "json" in ct else ""))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if cookie: self.send_header("Set-Cookie", cookie)
                self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                try: app.get(self)
                except (ConnectionError, OSError): pass
                except Exception as e:
                    try: self._send({"ok": False, "err": str(e)}, code=500)
                    except (ConnectionError, OSError): pass
            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n) or b"{}")
                    app.post(self, body)
                except (ConnectionError, OSError): pass
                except Exception as e:
                    try: self._send({"ok": False, "err": str(e)}, code=500)
                    except (ConnectionError, OSError): pass
        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0", port), H)
            self.srv.daemon_threads = True
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            self.port = port
            if open_browser:
                threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        except OSError:
            self.srv = None; self.port = None

    # ---- called by proctor loop --------------------------------------------
    def update(self, **kw): self.live.update(kw)

    def set_frame(self, frame):
        self._fn += 1
        if self._fn % 2: return
        h, w = frame.shape[:2]
        f = cv2.resize(frame, (560, int(560 * h / w)))
        ok, b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok: self.jpg = b.tobytes()

    def notify(self, event, hard):
        self.nid += 1
        self.notes.append({"id": self.nid, "t": datetime.now().strftime("%H:%M:%S"), "ev": event, "hard": hard,
                           "text": extras.VOICE_MSG.get(event, event.replace("_", " ").title())})

    def drain(self):
        out = []
        while not self.inq.empty(): out.append(self.inq.get())
        return out

    def begin(self, sid, user):
        self.sid, self.user, self.t0 = sid, user, time.time()
        self.answers, self.marks = {}, {}
        self.notes.clear(); self.msg = {"id": 0, "text": ""}
        self.pre = self.verified.pop(user["roll"], {})
        self._build_session()
        self.active = True

    def finish(self, reason, sid, cheats, score, grade, risk, rj, rh, total, idinfo=None):
        g = self.grade()
        v = extras._verdict(score, cheats)[0]
        res = {"sid": sid, "roll": self.user["roll"], "name": self.user["name"], "exam": self.exam["title"],
               "date": datetime.now().strftime("%d-%m-%Y %H:%M"), "correct": g["correct"], "total": len(self.order),
               "wrong": g["wrong"], "skipped": g["skipped"], "marks": g["marks"], "max_marks": g["max_marks"],
               "percent": g["percent"], "sections": g["sections"], "answered": len(self.answers), "integrity": score, "grade": grade, "violations": cheats,
               "verdict": v, "reason": reason, "duration": f"{int(total)//60}m {int(total)%60}s",
               "id_checks": (idinfo or {}).get("checks", 0), "id_ok": (idinfo or {}).get("ok", 0),
               "id_min": (idinfo or {}).get("min"), "pre_face": bool(self.pre.get("face")),
               "pre_voice": self.pre.get("voice") or "-"}
        self.results.append(res); _jsave("results.json", self.results)
        self.last_result[self.user["roll"]] = res
        self.active, self.pending_end, self.user = False, None, None
        return res

    def close(self):
        if self.srv: self.srv.shutdown()

    # ---- helpers -----------------------------------------------------------
    def _activate(self, pid):
        self.lib["active"] = pid
        self.exam = self.lib["papers"][pid]
        self.max_v, self.id_thr = int(self.exam["max_violations"]), float(self.exam["id_threshold"])
        _jsave("papers.json", self.lib)

    def _build_session(self):
        rnd, qs, ex = random.Random(), self.exam["questions"], self.exam
        if ex.get("shuffle"):
            secs = {}
            for i, q in enumerate(qs): secs.setdefault(q["section"], []).append(i)
            order = []
            for v in secs.values(): rnd.shuffle(v); order += v
        else:
            order = list(range(len(qs)))
        self.order, self.optmap, pub = order, [], []
        for oi in order:
            q = qs[oi]; n = len(q.get("options", [])); perm = list(range(n))
            if ex.get("shuffle") and q["type"] in ("single", "multi"): rnd.shuffle(perm)
            self.optmap.append(perm)
            pq = {"type": q["type"], "q": q["q"], "marks": q["marks"], "section": q["section"]}
            if q.get("code"): pq["code"] = q["code"]
            if n: pq["options"] = [q["options"][p] for p in perm]
            pub.append(pq)
        self.pub = {"title": ex["title"], "minutes": ex["minutes"], "max": self.max_v,
                    "negative": ex.get("negative", 0), "questions": pub}

    def grade(self):
        ex, neg = self.exam, self.exam.get("negative", 0.0)
        got = mx = 0.0; ok_n = bad_n = skip_n = 0; secs = {}
        for i, oi in enumerate(self.order):
            q = ex["questions"][oi]; m = q["marks"]; mx += m
            sc = secs.setdefault(q["section"], [0.0, 0.0]); sc[1] += m
            a, t = self.answers.get(i), q["type"]
            if a is None or a == "" or a == []: skip_n += 1; continue
            try:
                if t in ("single", "tf"): ok = self.optmap[i][int(a)] == q["answer"]
                elif t == "multi": ok = sorted(self.optmap[i][int(x)] for x in a) == q["answer"]
                elif t == "num": ok = abs(float(str(a).strip()) - q["answer"]) <= max(q.get("tol", 0), 1e-9)
                else: ok = str(a).strip().lower() in q["answer"]
            except Exception: ok = False
            if ok: got += m; sc[0] += m; ok_n += 1
            else:
                bad_n += 1
                if t in ("single", "tf", "multi"): got -= neg * m; sc[0] -= neg * m
        got = max(got, 0.0)
        return {"marks": round(got, 2), "max_marks": round(mx, 2), "percent": round(100 * got / mx, 1) if mx else 0,
                "correct": ok_n, "wrong": bad_n, "skipped": skip_n,
                "sections": {k: [round(max(v[0], 0), 2), round(v[1], 2)] for k, v in secs.items()}}

    def remaining(self):
        return max(0, int(self.exam["minutes"] * 60 - (time.time() - self.t0))) if self.active else 0

    def profile(self, u):
        return {k: u.get(k, "") for k in ("branch", "year", "email", "phone")}

    def stud_row(self, v):
        r = v["roll"]
        return {"name": v["name"], "roll": r, **self.profile(v), "photos": len(self.fid.photos(r)),
                "enrolled": self.fid.enrolled(r), "tests": sum(1 for x in self.results if x["roll"] == r)}

    @staticmethod
    def spoken_digits(text):
        W = {"zero": 0, "oh": 0, "one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "ate": 8, "nine": 9}
        return "".join(t if t.isdigit() else str(W.get(t, "")) for t in re.findall(r"[a-z]+|\d+", text.lower()))

    def wait(self, cond, sec):
        t = time.time()
        while time.time() - t < sec and not cond(): time.sleep(0.1)
        return cond()

    # ---- GET ---------------------------------------------------------------
    def get(self, h):
        path, _, qs = h.path.partition("?")
        q = dict(x.split("=", 1) for x in qs.split("&") if "=" in x)
        u = h._user()
        if path == "/":
            return h._send(PAGE.encode(), "text/html")
        if path == "/video":
            if not u: return h._send({"ok": False}, code=401)
            h.send_response(200)
            h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            h.send_header("Cache-Control", "no-store"); h.send_header("Connection", "close"); h.close_connection = True
            h.end_headers()
            while not self.stop:
                if self.jpg:
                    h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(self.jpg))
                    h.wfile.write(self.jpg); h.wfile.write(b"\r\n")
                time.sleep(0.12)
            return
        if path == "/api/me":
            if not u: return h._send({"ok": False})
            mine = [r for r in self.results if r["roll"] == u["roll"]]
            return h._send({"ok": True, "name": u["name"], "roll": u["roll"], "role": u["role"], "attempts": mine,
                            "profile": self.profile(u), "photos": len(self.fid.photos(u["roll"])),
                            "fid": {"ready": self.fid.ready, "err": self.fid.err},
                            "exam": {"title": self.exam["title"], "minutes": self.exam["minutes"], "n": len(self.exam["questions"]),
                                     "max": self.max_v, "tries": self.exam.get("attempts", 3),
                                     "req_id": bool(self.exam.get("require_identity")), "voice": bool(self.exam.get("voice_check")),
                                     "marks": round(sum(x["marks"] for x in self.exam["questions"]), 2),
                                     "negative": self.exam.get("negative", 0), "has_code": bool(self.exam.get("access_code")),
                                     "sections": sorted({x["section"] for x in self.exam["questions"]})}})
        if not u: return h._send({"ok": False, "err": "login"}, code=401)
        if path == "/api/state":
            since = int(q.get("since", 0))
            mine = self.active and self.user and self.user["roll"] == u["roll"]
            return h._send({"ok": True, "active": bool(mine), "busy": self.active and not mine,
                            "remaining": self.remaining() if mine else 0, "violations": self.live["cheats"] if mine else 0,
                            "max": self.max_v, "cls": self.live["cls"], "camera": bool(self.jpg),
                            "notes": [n for n in self.notes if n["id"] > since and n["hard"]] if mine else [],
                            "nid": self.nid, "msg": self.msg if mine else {"id": 0, "text": ""}})
        if path == "/api/exam":
            if not (self.active and self.user["roll"] == u["roll"]): return h._send({"ok": False})
            return h._send({"ok": True, "exam": self.pub, "remaining": self.remaining(),
                            "answers": self.answers, "marks": self.marks})
        if path == "/api/result":
            return h._send({"ok": True, "res": self.last_result.get(u["roll"])})
        if path.startswith("/api/photo/"):
            _, _, _, roll, n = path.split("/")[:5] if path.count("/") >= 4 else (0, 0, 0, "", "1")
            roll = re.sub(r"[^a-z0-9_.\-]", "", roll.lower())
            if u["role"] != "admin" and u["roll"] != roll: return h._send(b"", "text/plain", 403)
            ps = self.fid.photos(roll); i = max(1, int(re.sub(r"\D", "", n) or 1))
            f = next((p for p in ps if p.endswith(f"_{i}.jpg")), None)
            if not f: return h._send(b"", "text/plain", 404)
            return h._send(open(f, "rb").read(), "image/jpeg")
        if path == "/api/verify/code":
            self.codes[u["roll"]] = "".join(secrets.choice("0123456789") for _ in range(4))
            return h._send({"ok": True, "code": self.codes[u["roll"]]})
        if u["role"] != "admin": return h._send({"ok": False, "err": "admin only"}, code=403)
        if path == "/api/admin/state":
            d = dict(self.live)
            d.update(ok=True, active=self.active, student=self.user, remaining=self.remaining(), max=self.max_v,
                     notes=list(self.notes)[-30:], camera=bool(self.jpg), nid=self.nid,
                     answered=len(self.answers), total=len(self.exam["questions"]), sim=(self.live.get("mode") == "simulator"),
                     fid={"ready": self.fid.ready, "err": self.fid.err})
            return h._send(d)
        if path == "/api/admin/students":
            return h._send({"ok": True, "fid": {"ready": self.fid.ready, "err": self.fid.err},
                            "students": [self.stud_row(v) for v in self.users.values() if v["role"] == "student"]})
        if path == "/api/admin/results":
            return h._send({"ok": True, "results": self.results[::-1]})
        if path == "/api/admin/papers":
            return h._send({"ok": True, "active": self.lib["active"], "papers": [
                {"id": k, "title": v["title"], "n": len(v["questions"]), "minutes": v["minutes"]} for k, v in self.lib["papers"].items()]})
        if path == "/api/admin/exam":
            pid = re.sub(r"[^a-z0-9_\-]", "", q.get("id", "")) or self.lib["active"]
            return h._send({"ok": True, "id": pid, "exam": self.lib["papers"].get(pid, self.exam)})
        if path == "/api/admin/export":
            o = io.StringIO(); w = csv.writer(o)
            w.writerow(["Date", "Roll", "Name", "Correct", "Total", "Marks", "MaxMarks", "Percent", "Integrity", "Violations", "Verdict", "Reason", "IdentityChecks", "IdentityOK"])
            for r in self.results:
                w.writerow([r["date"], r["roll"], r["name"], r["correct"], r["total"], r.get("marks", ""), r.get("max_marks", ""), r.get("percent", ""), r["integrity"], r["violations"], r["verdict"], r["reason"], r.get("id_checks", 0), r.get("id_ok", 0)])
            return h._send(o.getvalue().encode("utf-8-sig"), "text/csv")
        if path.startswith("/report/"):
            sid = re.sub(r"[^0-9_]", "", path[8:])
            p = f"logs/report_{sid}.html"
            if os.path.exists(p):
                return h._send(open(p, "rb").read(), "text/html")
            return h._send(b"Report not found", "text/plain", 404)
        h._send({"ok": False}, code=404)

    # ---- POST --------------------------------------------------------------
    def post(self, h, b):
        path, u = h.path, h._user()
        if path == "/api/register":
            name, roll, pw = b.get("name", "").strip(), b.get("roll", "").strip().lower(), b.get("pw", "")
            if len(name) < 2 or not re.fullmatch(r"[a-z0-9_.\-]{2,30}", roll) or len(pw) < 4:
                return h._send({"ok": False, "err": "Name 2+ chars, ID letters/numbers only, password 4+ chars"})
            if roll in self.users: return h._send({"ok": False, "err": "This ID is already registered"})
            s = secrets.token_hex(8)
            self.users[roll] = {"name": name, "roll": roll, "salt": s, "hash": _hash(pw, s), "role": "student"}
            _jsave("users.json", self.users)
            return h._send({"ok": True})
        if path == "/api/login":
            roll = b.get("roll", "").strip().lower(); x = self.users.get(roll)
            if not x or _hash(b.get("pw", ""), x["salt"]) != x["hash"]:
                return h._send({"ok": False, "err": "Wrong ID or password"})
            tok = secrets.token_hex(16); self.sessions[tok] = roll
            return h._send({"ok": True, "role": x["role"]}, cookie=f"sp_token={tok}; Path=/; HttpOnly; SameSite=Lax")
        if path == "/api/logout":
            return h._send({"ok": True}, cookie="sp_token=; Path=/; Max-Age=0")
        if not u: return h._send({"ok": False, "err": "login"}, code=401)
        mine = self.active and self.user and self.user["roll"] == u["roll"]
        if path == "/api/start":
            if u["role"] != "student": return h._send({"ok": False, "err": "Admin cannot take exam"})
            if self.active: return h._send({"ok": mine, "err": "Another student is taking the exam"})
            tries = sum(1 for r in self.results if r["roll"] == u["roll"])
            if tries >= self.exam.get("attempts", 3): return h._send({"ok": False, "err": "No attempts left"})
            if not self.jpg: return h._send({"ok": False, "err": "Camera not ready yet. Wait a few seconds."})
            ac = self.exam.get("access_code", "")
            if ac and str(b.get("code", "")).strip().lower() != ac.lower():
                return h._send({"ok": False, "err": "Wrong exam access code"})
            if self.exam.get("require_identity"):
                v = self.verified.get(u["roll"])
                if not self.fid.ready: return h._send({"ok": False, "err": self.fid.err})
                if not self.fid.enrolled(u["roll"]):
                    return h._send({"ok": False, "err": "Your photo is not enrolled. Ask your teacher to upload it."})
                if not v or not v.get("face") or time.time() - v["t"] > 600:
                    return h._send({"ok": False, "err": "Identity not verified. Please verify your face first."})
                if self.exam.get("voice_check") and v.get("voice") not in ("ok", "skipped", "typed"):
                    return h._send({"ok": False, "err": "Voice check not completed."})
            self.pending_start = u
            ok = self.wait(lambda: self.active, 8)
            return h._send({"ok": ok, "err": "" if ok else "Proctor engine not responding"})
        if path == "/api/verify/face":
            roll = u["roll"]
            if not self.fid.ready: return h._send({"ok": False, "err": self.fid.err})
            if not self.fid.enrolled(roll): return h._send({"ok": False, "err": "Your photo is not enrolled. Ask your teacher."})
            hits = best = none_ = multi = 0; best = -1.0
            for _ in range(4):
                img = None if self.raw is None else self.raw.copy()
                f, n, w = self.fid.embed(img) if img is not None else (None, 0, 0)
                if f is None: none_ += 1
                elif n > 1: multi += 1
                else:
                    sc = max(self.fid._cos(f, e) for e in self.fid.feats[roll]); best = max(best, sc)
                    hits += sc >= self.id_thr
                time.sleep(0.4)
            ok = hits >= 3
            v = self.verified.setdefault(roll, {"voice": None}); v.update(t=time.time(), face=ok, score=round(best, 3))
            msg = ("Face verified" if ok else "Face not visible. Look at the camera in good light." if none_ >= 2 else
                   "Only ONE person must be in front of the camera." if multi >= 2 else "Face does not match the enrolled photo.")
            return h._send({"ok": True, "match": ok, "score": round(best, 3), "need": self.id_thr, "msg": msg})
        if path == "/api/verify/voice":
            roll = u["roll"]; v = self.verified.setdefault(roll, {"face": False, "voice": None, "t": time.time()})
            if b.get("skip"):
                v.update(voice="skipped", t=time.time()); return h._send({"ok": True, "match": True, "skipped": True})
            code = self.codes.get(roll, "x")
            if b.get("typed") is not None:
                ok = str(b["typed"]).strip() == code
                if ok: v.update(voice="typed", t=time.time())
                return h._send({"ok": True, "match": ok, "heard": "typed"})
            ok = any(code in self.spoken_digits(str(t)) for t in b.get("texts", []))
            if ok: v.update(voice="ok", t=time.time())
            return h._send({"ok": True, "match": ok, "heard": str((b.get("texts") or [""])[0])[:60]})
        if path == "/api/answer" and mine:
            i = int(b["i"])
            if not 0 <= i < len(self.order): return h._send({"ok": False})
            t, c = self.exam["questions"][self.order[i]]["type"], b.get("c")
            if c is None or c == "" or c == []: self.answers.pop(i, None)
            elif t in ("single", "tf"): self.answers[i] = int(c)
            elif t == "multi": self.answers[i] = sorted({int(x) for x in c})
            else: self.answers[i] = str(c)[:80]
            if b.get("m"): self.marks[i] = 1
            else: self.marks.pop(i, None)
            return h._send({"ok": True})
        if path == "/api/violation" and mine:
            t = b.get("type")
            lv = {"tab_switch": "hard", "fullscreen_exit": "hard", "copy_paste": "soft", "right_click": "soft",
                  "mic_talking": "hard", "mic_noise": "soft", "multi_monitor": "hard", "blocked_key": "soft"}.get(t)
            if lv: self.inq.put((lv, t, "browser: " + t.replace("_", " ")))
            return h._send({"ok": True})
        if path == "/api/submit":
            if mine:
                self.pending_end = "submitted"
                self.wait(lambda: not self.active, 15)
            return h._send({"ok": True, "res": self.last_result.get(u["roll"])})
        if u["role"] != "admin": return h._send({"ok": False, "err": "admin only"}, code=403)
        if path.startswith("/api/admin/") and path.rsplit("/", 1)[1] in ("profile", "photo", "photo_del", "passwd"):
            pw_ok = _hash(b.get("admin_pw", b.get("old", "")), u["salt"]) == u["hash"]
            if not pw_ok: return h._send({"ok": False, "err": "Admin password is wrong"})
            act = path.rsplit("/", 1)[1]
            if act == "passwd":
                if len(b.get("new", "")) < 5: return h._send({"ok": False, "err": "New password must be 5+ characters"})
                s2 = secrets.token_hex(8); u["salt"], u["hash"] = s2, _hash(b["new"], s2)
                _jsave("users.json", self.users); return h._send({"ok": True, "msg": "Password changed"})
            roll = str(b.get("roll", "")).lower(); t = self.users.get(roll)
            if not t or t["role"] != "student": return h._send({"ok": False, "err": "Student not found"})
            if act == "profile":
                if len(b.get("name", "").strip()) >= 2: t["name"] = b["name"].strip()[:60]
                for k in ("branch", "year", "email", "phone"): t[k] = str(b.get(k, ""))[:60].strip()
                if b.get("newpw"):
                    if len(b["newpw"]) < 4: return h._send({"ok": False, "err": "Student password must be 4+ characters"})
                    s2 = secrets.token_hex(8); t["salt"], t["hash"] = s2, _hash(b["newpw"], s2)
                _jsave("users.json", self.users); return h._send({"ok": True, "msg": "Details saved"})
            if act == "photo_del":
                self.fid.del_photo(roll, b.get("n", 1)); return h._send({"ok": True, "msg": "Photo deleted"})
            img = None
            if b.get("from_camera"):
                img = None if self.raw is None else self.raw.copy()
            else:
                try:
                    data = base64.b64decode(str(b.get("image", "")).split(",")[-1])
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                except Exception: img = None
            if img is None: return h._send({"ok": False, "err": "Could not read the image"})
            ok, msg = self.fid.add_photo(roll, img)
            return h._send({"ok": ok, "err": "" if ok else msg, "msg": msg})
        if path == "/api/admin/sim":
            if self.sim_cb: self.sim_cb(b.get("k"))
            return h._send({"ok": True})
        if path == "/api/admin/msg":
            self.msg = {"id": self.msg["id"] + 1, "text": str(b.get("text", ""))[:200]}
            return h._send({"ok": True})
        if path == "/api/admin/end":
            if self.active: self.pending_end = "ended by teacher"
            return h._send({"ok": True})
        if path == "/api/admin/shutdown":
            self.stop = True; return h._send({"ok": True})
        if path.startswith("/api/admin/paper_"):
            act = path.rsplit("_", 1)[1]; pid = re.sub(r"[^a-z0-9_\-]", "", str(b.get("id", "")).lower())
            P = self.lib["papers"]
            if act == "save":
                try: e = normalize_exam(validate_exam(b["exam"]))
                except Exception: return h._send({"ok": False, "err": "Invalid paper (check title, minutes, question types, options and answers)"})
                if self.active and pid == self.lib["active"]: return h._send({"ok": False, "err": "Cannot edit the active paper while an exam is running"})
                if pid not in P: return h._send({"ok": False, "err": "Unknown paper"})
                P[pid] = e
                self._activate(self.lib["active"]); return h._send({"ok": True})
            if act == "new":
                if not re.fullmatch(r"[a-z0-9_\-]{2,20}", pid) or pid in P: return h._send({"ok": False, "err": "Use a new id (2-20 letters/numbers)"})
                e = copy.deepcopy(self.exam); e["title"] = str(b.get("title") or pid)[:80]; P[pid] = e
                self._activate(self.lib["active"]); return h._send({"ok": True, "id": pid})
            if act == "select":
                if self.active: return h._send({"ok": False, "err": "An exam is running right now"})
                if pid not in P: return h._send({"ok": False, "err": "Unknown paper"})
                self._activate(pid); return h._send({"ok": True})
            if act == "del":
                if pid == self.lib["active"] or pid not in P or len(P) < 2:
                    return h._send({"ok": False, "err": "Cannot delete the active (or only) paper"})
                del P[pid]; self._activate(self.lib["active"]); return h._send({"ok": True})
        h._send({"ok": False}, code=404)


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Smart Proctor PRO</title>
<style>
:root{--bg:#0b1020;--p:#131a2e;--p2:#1a2340;--b:#26325a;--t:#e8ecf7;--m:#8d9bc4;--a:#4f8cff;--ok:#2ecc71;--w:#f5b93b;--r:#ff5468}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,Arial;background:var(--bg);color:var(--t)}
button,input,textarea,select{font:inherit}a{color:var(--a)}
.btn{background:var(--a);color:#fff;border:0;border-radius:10px;padding:10px 18px;cursor:pointer;font-weight:600}
.btn:hover{filter:brightness(1.1)}.btn.g{background:var(--p2);border:1px solid var(--b);color:var(--t)}.btn.r{background:var(--r)}.btn.s{padding:6px 12px;font-size:13px}
.btn:disabled{opacity:.5;cursor:not-allowed}
input,textarea,select{width:100%;background:var(--p2);border:1px solid var(--b);color:var(--t);border-radius:10px;padding:11px;margin:5px 0 12px;outline:0}
input:focus,textarea:focus{border-color:var(--a)}label{font-size:13px;color:var(--m)}
.card{background:var(--p);border:1px solid var(--b);border-radius:16px;padding:18px}
.auth{display:grid;grid-template-columns:1.1fr 1fr;min-height:100vh}
.hero{padding:60px;background:radial-gradient(900px 500px at 10% 10%,#1d3a8a55,transparent),linear-gradient(160deg,#0f1a3d,#0b1020);display:flex;flex-direction:column;justify-content:center}
.hero h1{font-size:40px;margin:0 0 6px}.hero h1 span{color:var(--a)}.hero li{margin:10px 0;color:#c5d0ee;list-style:none}
.hero li:before{content:"\2713  ";color:var(--ok);font-weight:700}
.formw{display:flex;align-items:center;justify-content:center;padding:30px}.form{width:100%;max-width:400px}
.tabs{display:flex;gap:6px;margin-bottom:16px}.tabs button{flex:1}
.err{color:var(--r);font-size:13px;min-height:18px;margin-bottom:8px}
.top{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:var(--p);border-bottom:1px solid var(--b)}
.logo{font-weight:800;font-size:18px}.logo span{color:var(--a)}
.wrap{max-width:1100px;margin:24px auto;padding:0 16px}
.grid{display:grid;gap:14px}.g2{grid-template-columns:1.3fr 1fr}.g4{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.kpi small{color:var(--m);display:block}.kpi b{font-size:26px}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:700;font-size:13px}
.OK{background:#1e5e3a;color:#8ff0b6}.WARN{background:#6b5314;color:#ffd873}.CHEAT{background:#7a1f2c;color:#ff9aa8}
video,img.cam{width:100%;border-radius:12px;background:#000;display:block}
.side{width:210px;background:var(--p);border-right:1px solid var(--b);min-height:calc(100vh - 53px);padding:14px}
.side a{display:block;padding:10px 12px;border-radius:10px;color:var(--t);text-decoration:none;cursor:pointer;margin-bottom:4px}
.side a.on,.side a:hover{background:var(--p2);color:#fff}
table{width:100%;border-collapse:collapse;font-size:14px}th{color:var(--m);text-align:left;font-weight:600}td,th{padding:9px 8px;border-bottom:1px solid var(--b)}
.feed{max-height:260px;overflow:auto;font-size:13px}.feed div{padding:6px 0;border-bottom:1px solid var(--b)}.feed .h{color:#ff9aa8}.feed .s{color:#ffd873}
.ex{display:grid;grid-template-columns:1fr 290px;gap:14px;max-width:1200px;margin:16px auto;padding:0 14px}
.opt{display:block;padding:14px;border:1px solid var(--b);border-radius:12px;margin:10px 0;cursor:pointer;background:var(--p2)}
.opt:hover{border-color:var(--a)}.opt.on{border-color:var(--a);background:#1b3270}
.pal{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.pal button{padding:9px 0;border-radius:8px;border:1px solid var(--b);background:var(--p2);color:var(--t);cursor:pointer}
.pal .ans{background:#1e5e3a}.pal .mk{background:#6b5314}.pal .cur{outline:2px solid var(--a)}
.timer{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}
#toast{position:fixed;top:70px;right:20px;z-index:9;display:flex;flex-direction:column;gap:8px}
.tst{background:#7a1f2c;border:1px solid var(--r);padding:12px 16px;border-radius:12px;max-width:340px;box-shadow:0 6px 24px #0008}
.tst.info{background:#173a78;border-color:var(--a)}
#modal{position:fixed;inset:0;background:#000b;display:none;align-items:center;justify-content:center;z-index:10}
#modal .card{max-width:420px}
@media(max-width:800px){.auth,.g2,.ex{grid-template-columns:1fr}.hero{display:none}.side{width:100%;min-height:0;display:flex;overflow:auto}}
</style></head><body><div id=app></div><div id=toast></div><div id=modal><div class=card id=mbox></div></div>
<script>
const $=s=>document.querySelector(s);let finalWarned0=0;let me=null,exam=null,cur=0,ans={},mk={},since=0,pol=null,lastMsg=0,inExam=false,ended=false,live=null;
async function api(p,b){try{const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return await r.json()}catch(e){return{ok:false,err:'Server not reachable'}}}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function toast(t,cls){const d=document.createElement('div');d.className='tst '+(cls||'');d.textContent=t;$('#toast').appendChild(d);setTimeout(()=>d.remove(),6000)}
function speak(t){try{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.rate=1;speechSynthesis.speak(u)}catch(e){}}
function modal(h,w){$('#mbox').style.maxWidth=(w||420)+'px';$('#mbox').style.width='100%';$('#mbox').innerHTML=h;$('#modal').style.display='flex'}function closeM(){$('#modal').style.display='none'}
async function init(){me=await api('/api/me');if(!me.ok)return authView();me.role=='admin'?adminView('live'):homeView()}
/* ---------- AUTH ---------- */
function authView(tab){tab=tab||'login';const reg=tab=='reg';
$('#app').innerHTML=`<div class=auth><div class=hero><div class=logo style="font-size:22px">&#128737; Smart<span>Proctor</span> PRO</div><h1>Secure online exams,<br><span>watched by AI + IoT</span></h1>
<ul><li>Face, gaze &amp; head-pose AI proctoring</li><li>Phone / device detection (YOLO)</li><li>IoT sensor fusion: camera-tamper &amp; spoof detection</li><li>Live integrity score &amp; instant warnings</li><li>Auto evidence photos &amp; teacher report</li></ul></div>
<div class=formw><div class=form><div class=tabs><button class="btn ${reg?'g':''}" onclick="authView('login')">Login</button><button class="btn ${reg?'':'g'}" onclick="authView('reg')">Register</button></div>
${reg?'<label>Full name</label><input id=n placeholder="Your name">':''}<label>${reg?'Student ID / Roll no':'ID / Roll no'}</label><input id=r placeholder="e.g. 21cs045" autocomplete=username>
<label>Password</label><input id=p type=password onkeydown="if(event.key=='Enter')go()" autocomplete=current-password><div class=err id=e></div>
<button class=btn style="width:100%" onclick="go()">${reg?'Create account':'Login'}</button>
<p style="color:var(--m);font-size:12px;margin-top:16px">${reg?'':'Teacher login: admin / admin123'}</p></div></div></div>`;
window.go=async()=>{const b={roll:$('#r').value,pw:$('#p').value};if(reg)b.name=$('#n').value;const r=await api(reg?'/api/register':'/api/login',b);
if(!r.ok)return $('#e').textContent=r.err;if(reg){authView('login');setTimeout(()=>{$('#e').style.color='#2ecc71';$('#e').textContent='Registered! Now login.'},50)}else init()}}
async function logout(){await api('/api/logout',{});clearInterval(pol);location.reload()}
/* ---------- STUDENT HOME ---------- */
async function homeView(){me=await api('/api/me');const e=me.exam,left=e.tries-me.attempts.length;
const last=me.attempts[me.attempts.length-1];
$('#app').innerHTML=`<div class=top><div class=logo>&#128737; Smart<span>Proctor</span></div><div>${esc(me.name)} (${esc(me.roll)}) &nbsp;<button class="btn g s" onclick=logout()>Logout</button></div></div>
<div class=wrap><div class="grid g2"><div class=card><h2 style="margin-top:0">${esc(e.title)}</h2>
<div class="grid g4" style="margin:14px 0"><div class="card kpi"><small>Questions</small><b>${e.n}</b></div><div class="card kpi"><small>Duration</small><b>${e.minutes} min</b></div><div class="card kpi"><small>Attempts left</small><b>${Math.max(0,left)}</b></div><div class="card kpi"><small>Total marks</small><b>${e.marks}</b></div></div>
<h3>Rules</h3><ul style="line-height:1.9;color:#c5d0ee"><li>Sit in front of the camera, face clearly visible, good light.</li><li>Do not look away, talk, use phone/other devices or let anyone else in.</li><li>Stay in fullscreen. Switching tab / window is recorded as a violation.</li><li>After <b>${e.max}</b> violations the exam auto-submits.</li><li>Warnings appear on screen and are spoken aloud.</li><li>Microphone is monitored for talking/noise - allow mic access and stay silent.</li><li>Extra monitors, copy/paste and shortcuts (F12, Ctrl+C...) are blocked and recorded.</li>${e.negative?`<li>Negative marking: -${e.negative} x marks for each wrong MCQ.</li>`:''}</ul>${e.has_code?'<label>Exam access code (ask your teacher)</label><input id=ac style="max-width:260px" autocomplete=off>':''}
<label style="display:flex;gap:8px;align-items:center;color:var(--t)"><input type=checkbox id=ag style="width:auto;margin:0" onchange="$('#sb').disabled=!this.checked||${left<=0}"> I agree to be monitored during the exam</label><div class=err id=e style="margin-top:10px"></div>
<button class=btn id=sb disabled onclick=beginFlow()>Start Exam</button></div>
<div class=card><div style="display:flex;gap:14px;align-items:center;margin-bottom:14px">${me.photos?`<img src="/api/photo/${me.roll}/1?t=${Date.now()}" style="width:84px;height:84px;object-fit:cover;border-radius:14px;border:2px solid var(--ok)">`:`<div style="width:84px;height:84px;border-radius:14px;background:var(--p2);display:flex;align-items:center;justify-content:center;font-size:34px">&#128100;</div>`}
<div><b style="font-size:17px">${esc(me.name)}</b><div style="color:var(--m);font-size:13px">ID: ${esc(me.roll)}${me.profile.branch?' &middot; '+esc(me.profile.branch):''}${me.profile.year?' &middot; '+esc(me.profile.year):''}</div>
<div style="font-size:13px;margin-top:4px">${me.photos?'<span style="color:var(--ok)">&#10003; Photo enrolled for face verification</span>':'<span style="color:var(--w)">&#9888; Photo not enrolled - ask your teacher</span>'}</div></div></div>
<h3 style="margin-top:0">System check</h3><img class=cam src=/video alt="camera"><p id=chk style="color:var(--m)">Checking camera...</p>
${last?`<hr style="border-color:var(--b)"><h3>Last result</h3><p>Score <b>${last.correct}/${last.total}</b> &nbsp; Integrity <b>${last.integrity}</b> &nbsp; <span class="pill ${last.verdict=='TRUSTED'?'OK':last.verdict=='SUSPICIOUS'?'CHEAT':'WARN'}">${last.verdict}</span></p>`:''}</div></div></div>`;
const c=async()=>{const s=await api('/api/state');if(!$('#chk'))return;if(s.active&&!inExam)return resume();$('#chk').innerHTML=s.busy?'&#9888; Another student is taking the exam.':s.camera?'<span style="color:var(--ok)">&#9679; Camera &amp; proctor engine ready</span>':'&#9888; Camera starting...';};
c();clearInterval(pol);pol=setInterval(c,2000)}
let acode='';const showErr=t=>{$('#e')?$('#e').textContent=t:toast(t)};
async function micPerm(){try{const m=await navigator.mediaDevices.getUserMedia({audio:true});m.getTracks().forEach(t=>t.stop())}catch(e){}}
async function startExam(){if($('#sb'))$('#sb').disabled=true;if($('#gob'))$('#gob').disabled=true;try{await document.documentElement.requestFullscreen()}catch(e){}
const r=await api('/api/start',{code:acode});if(!r.ok){showErr(r.err||'Cannot start');if($('#sb'))$('#sb').disabled=false;if($('#gob'))$('#gob').disabled=false;if(r.err&&r.err.includes('code'))homeView();return}resume()}
/* ---------- IDENTITY VERIFICATION ---------- */
async function beginFlow(){$('#e').textContent='';acode=$('#ac')?$('#ac').value:'';if(me.exam.has_code&&!acode.trim())return $('#e').textContent='Enter the exam access code';await micPerm();if(!me.exam.req_id)return startExam();
if(!me.fid.ready)return $('#e').textContent=me.fid.err||'Face engine loading, wait...';
if(!me.photos)return $('#e').textContent='Your photo is not enrolled. Ask your teacher to upload it.';verifyView()}
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;let faceOk=false,voiceOk=false,vcode='';
async function verifyView(){faceOk=false;voiceOk=!me.exam.voice;clearInterval(pol);
$('#app').innerHTML=`<div class=top><div class=logo>&#128737; Identity verification</div><button class="btn g s" onclick=homeView()>Cancel</button></div>
<div class=wrap style="max-width:820px"><div class="grid g2"><div class=card><img class=cam src=/video></div><div class=card>
<h3 style="margin-top:0">Step 1 &middot; Face</h3><p style="color:var(--m)">Sit straight, good light, only you in frame.</p><button class=btn id=vf onclick=doFace()>Verify my face</button><div id=fr style="margin:10px 0;min-height:22px"></div>
${me.exam.voice?`<h3>Step 2 &middot; Voice</h3><p style="color:var(--m)">Say these digits clearly:</p><div id=vc style="font-size:38px;letter-spacing:10px;font-weight:800;color:var(--a)">----</div><button class="btn g" id=vb onclick=doVoice() disabled>&#127908; Speak the code</button><div id=vr style="margin:10px 0;min-height:22px"></div>
<label style="font-size:13px;color:var(--m)">Mic not working? Type the digits instead:</label><div style="display:flex;gap:6px;margin-top:4px"><input id=vt placeholder="4 digits" maxlength=4 inputmode=numeric style="margin:0" onkeydown="if(event.key=='Enter')doTyped()"><button class="btn g s" id=vtb onclick=doTyped() disabled>Verify</button></div>`:''}
<button class=btn id=gob style="width:100%;margin-top:8px" disabled onclick=startExam()>Begin exam</button></div></div></div>`;
if(me.exam.voice){const c=await api('/api/verify/code');vcode=c.code;$('#vc').textContent=c.code}}
function gate(){$('#gob').disabled=!(faceOk&&voiceOk);if($('#vb'))$('#vb').disabled=!faceOk||voiceOk;if($('#vtb'))$('#vtb').disabled=!faceOk||voiceOk}
async function doFace(){$('#vf').disabled=true;$('#fr').innerHTML='Scanning... keep looking at the camera';speak('Please look at the camera');
const r=await api('/api/verify/face',{});$('#vf').disabled=false;
if(!r.ok){$('#fr').innerHTML='<span style="color:var(--r)">'+esc(r.err)+'</span>';return}
faceOk=r.match;$('#fr').innerHTML=r.match?`<span style="color:var(--ok)">&#10003; Verified &middot; similarity ${r.score}</span>`:`<span style="color:var(--r)">&#10007; ${esc(r.msg)} (similarity ${r.score}, need ${r.need})</span>`;
if(r.match){speak('Face verified')}gate()}
async function doTyped(){const r=await api('/api/verify/voice',{typed:$('#vt').value});voiceOk=r.match;
$('#vr').innerHTML=r.match?'<span style="color:var(--ok)">&#10003; Code verified (typed)</span>':'<span style="color:var(--r)">&#10007; Wrong code. Check the digits above.</span>';if(r.match)speak('Code verified. You may begin.');gate()}
async function doVoice(){$('#vb').disabled=true;
if(!SR){await api('/api/verify/voice',{skip:true});voiceOk=true;$('#vr').innerHTML='<span style="color:var(--w)">Voice recognition not supported in this browser - skipped (use Chrome/Edge)</span>';return gate()}
$('#vr').textContent='Listening...';
const texts=await new Promise(res=>{const r=new SR();r.lang='en-IN';r.maxAlternatives=4;r.onresult=e=>res([...e.results[0]].map(a=>a.transcript));r.onerror=e=>res({err:e.error});r.onend=()=>res([]);r.start()});
if(texts.err){if(['not-allowed','network','service-not-allowed'].includes(texts.err)){await api('/api/verify/voice',{skip:true});voiceOk=true;$('#vr').innerHTML='<span style="color:var(--w)">Mic/voice service unavailable ('+texts.err+') - skipped</span>';return gate()}
$('#vr').innerHTML='<span style="color:var(--r)">Could not hear you ('+texts.err+'). Try again.</span>';$('#vb').disabled=false;return}
const r=await api('/api/verify/voice',{texts});voiceOk=r.match;
$('#vr').innerHTML=r.match?'<span style="color:var(--ok)">&#10003; Voice code matched</span>':`<span style="color:var(--r)">&#10007; Heard "${esc(r.heard||'nothing')}". Try again.</span>`;
if(r.match)speak('Voice verified. You may begin.');gate();if(!r.match)$('#vb').disabled=false}
/* ---------- EXAM ---------- */
async function resume(){const r=await api('/api/exam');if(!r.ok)return homeView();exam=r.exam;ans=r.answers||{};mk=r.marks||{};cur=0;since=0;inExam=true;ended=false;finalWarned=false;lastMM=0;clearInterval(pol);
$('#app').innerHTML=`<div class=top><div class=logo>${esc(exam.title)}</div><div class=timer id=tm>--:--</div><div>Violations <b id=vc>0</b>/${exam.max} &nbsp;<button class="btn s" onclick=confirmSubmit()>Submit</button></div></div>
<div class=ex><div class=card id=qa></div><div><div class=card style="margin-bottom:14px"><img class=cam src=/video><div style="margin-top:8px;font-size:13px;color:var(--m)"><span id=dot style="color:var(--ok)">&#9679;</span> Proctoring active</div></div>
<div class=card style="margin-bottom:14px"><div style="font-size:13px;color:var(--m)">&#127908; <span id=mic>Mic: starting...</span></div><div style="height:6px;background:var(--p2);border-radius:4px;margin-top:6px"><div id=mbar style="height:6px;width:0;background:var(--ok);border-radius:4px"></div></div></div>
<div class=card><b>Questions</b><div class=pal id=pal style="margin-top:10px"></div><p style="font-size:12px;color:var(--m)"><span style="color:#2ecc71">&#9632;</span> answered &nbsp;<span style="color:#f5b93b">&#9632;</span> review</p></div></div></div>`;
renderQ();examStart=Date.now();startMic();pol=setInterval(tick,1000);tick()}
let examStart=0,tt=null;
function renderPal(){$('#pal').innerHTML=exam.questions.map((_,i)=>`<button class="${ans[i]!==undefined?'ans':''} ${mk[i]?'mk':''} ${i==cur?'cur':''}" onclick="go2(${i})">${i+1}</button>`).join('')}
function renderQ(){const q=exam.questions[cur],a=ans[cur],t=q.type,mcq=['single','tf','multi'].includes(t);let body='';
if(t=='single'||t=='tf')body=q.options.map((o,i)=>`<div class="opt ${a===i?'on':''}" onclick="pick(${i})"><input type=radio style="width:auto;margin:0 10px 0 0;pointer-events:none" ${a===i?'checked':''}>${esc(o)}</div>`).join('');
else if(t=='multi')body='<small style="color:var(--m)">Select all that apply</small>'+q.options.map((o,i)=>{const on=Array.isArray(a)&&a.includes(i);return`<div class="opt ${on?'on':''}" onclick="tog(${i})"><input type=checkbox style="width:auto;margin:0 10px 0 0;pointer-events:none" ${on?'checked':''}>${esc(o)}</div>`}).join('');
else body=`<input id=ti ${t=='num'?'inputmode=decimal':''} placeholder="${t=='num'?'Type the number':'Type your answer'}" value="${esc(a===undefined?'':a)}" oninput="typ(this.value)" style="font-size:18px;margin-top:14px" autocomplete=off>`;
$('#qa').innerHTML=`<small style="color:var(--m)"><b style="color:var(--a)">${esc(q.section)}</b> &middot; Question ${cur+1} of ${exam.questions.length} &middot; ${q.marks} mark${q.marks>1?'s':''}${exam.negative&&mcq?' &middot; -'+(exam.negative*q.marks)+' if wrong':''}</small><h2 style="margin:8px 0 14px">${esc(q.q)}</h2>
${q.code?`<pre style="background:#0a0f1e;border:1px solid var(--b);border-radius:10px;padding:12px;overflow:auto;font-size:14px">${esc(q.code)}</pre>`:''}${body}
<div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap"><button class="btn g" onclick="go2(cur-1)" ${cur==0?'disabled':''}>&larr; Prev</button><button class="btn g" onclick="clr()">Clear</button>
<button class="btn g" onclick="rev()">${mk[cur]?'Unmark':'Mark for review'}</button><button class=btn onclick="go2(cur+1)" ${cur==exam.questions.length-1?'disabled':''}>Next &rarr;</button></div>`;renderPal()}
function save(){api('/api/answer',{i:cur,c:ans[cur]===undefined?null:ans[cur],m:!!mk[cur]})}
function pick(i){ans[cur]=i;save();renderQ()}function clr(){delete ans[cur];save();renderQ()}
function tog(i){let a=Array.isArray(ans[cur])?[...ans[cur]]:[];a.includes(i)?a.splice(a.indexOf(i),1):a.push(i);a.sort((x,y)=>x-y);a.length?ans[cur]=a:delete ans[cur];save();renderQ()}
function typ(v){v.trim()===''?delete ans[cur]:ans[cur]=v;renderPal();clearTimeout(tt);const k=cur;tt=setTimeout(()=>{if(k==cur)save()},500)}
function rev(){mk[cur]?delete mk[cur]:mk[cur]=1;save();renderQ()}function go2(n){clearTimeout(tt);if(ans[cur]!==undefined&&typeof ans[cur]=='string')save();cur=Math.max(0,Math.min(exam.questions.length-1,n));renderQ()}
/* ---- microphone monitor (talking / noise) ---- */
let micS=null,micC=null,micT=null,lastMM=0,finalWarned=false;
async function startMic(){try{micS=await navigator.mediaDevices.getUserMedia({audio:true});micC=new (window.AudioContext||window.webkitAudioContext)();const an=micC.createAnalyser();an.fftSize=1024;micC.createMediaStreamSource(micS).connect(an);
const buf=new Uint8Array(an.fftSize);let base=[],loud=[],nT=0,tT=0;$('#mic').textContent='Mic: calibrating (stay quiet 3s)...';
micT=setInterval(()=>{an.getByteTimeDomainData(buf);let sq=0;for(const v of buf){const x=(v-128)/128;sq+=x*x}const rms=Math.sqrt(sq/buf.length);if($('#mbar'))$('#mbar').style.width=Math.min(100,rms*500)+'%';
if(base.length<30){base.push(rms);if(base.length==30&&$('#mic'))$('#mic').textContent='Mic: monitoring';return}
const thr=Math.max(base.reduce((x,y)=>x+y)/base.length*3,0.05);loud.push(rms>thr?1:0);if(loud.length>60)loud.shift();const now=Date.now();
if(loud.length>=15&&loud.slice(-15).every(x=>x)&&now-nT>10000){nT=now;vio('mic_noise')}
if(loud.length>=60&&loud.reduce((x,y)=>x+y)/60>=0.5&&now-tT>20000){tT=now;vio('mic_talking')}},100)}catch(e){if($('#mic'))$('#mic').textContent='Mic: off (permission denied)'}}
function stopMic(){clearInterval(micT);try{micS&&micS.getTracks().forEach(t=>t.stop())}catch(e){}try{micC&&micC.close()}catch(e){}micS=micC=null}
function confirmSubmit(){const un=exam.questions.length-Object.keys(ans).length;modal(`<h3 style="margin-top:0">Submit exam?</h3><p>${un?`You have <b>${un}</b> unanswered question(s).`:'All questions answered.'}</p><div style="display:flex;gap:8px"><button class=btn onclick="closeM();finish()">Yes, submit</button><button class="btn g" onclick=closeM()>Go back</button></div>`)}
async function finish(){if(ended)return;ended=true;clearInterval(pol);const r=await api('/api/submit',{});stopMic();inExam=false;try{document.exitFullscreen()}catch(e){}resultView(r.res)}
function resultView(r){$('#app').innerHTML=`<div class=top><div class=logo>&#128737; Smart<span>Proctor</span></div><button class="btn g s" onclick=homeView()>Back</button></div>
<div class=wrap style="max-width:680px"><div class=card style="text-align:center"><h1 style="margin-top:0">Exam submitted</h1>${r?`<div class="grid g4"><div class="card kpi"><small>Marks</small><b>${r.marks}/${r.max_marks}</b></div><div class="card kpi"><small>Percent</small><b>${r.percent}%</b></div><div class="card kpi"><small>Integrity</small><b>${r.integrity}</b></div><div class="card kpi"><small>Violations</small><b>${r.violations}</b></div></div>
<p style="color:var(--m)">Correct ${r.correct} &middot; Wrong ${r.wrong} &middot; Skipped ${r.skipped}</p>
${Object.entries(r.sections||{}).map(([k,v])=>`<div style="text-align:left;margin:8px 0"><small>${esc(k)} - ${v[0]}/${v[1]}</small><div style="height:8px;background:var(--p2);border-radius:6px"><div style="height:8px;width:${v[1]?Math.max(0,v[0]/v[1]*100):0}%;background:var(--a);border-radius:6px"></div></div></div>`).join('')}
<p style="margin-top:16px"><span class="pill ${r.verdict=='TRUSTED'?'OK':r.verdict=='SUSPICIOUS'?'CHEAT':'WARN'}">${r.verdict}</span></p><p style="color:var(--m)">Ended: ${esc(r.reason)} &middot; ${r.duration}<br>Your proctoring report was sent to the teacher.</p>`:'<p>Submitted.</p>'}
<button class=btn onclick=homeView()>Done</button></div></div>`}
async function tick(){const s=await api('/api/state?since='+since);if(!s.ok)return;
if(!s.active){if(inExam&&!ended){ended=true;clearInterval(pol);const r=await api('/api/result');stopMic();inExam=false;try{document.exitFullscreen()}catch(e){}resultView(r.res)}return}
const m=Math.floor(s.remaining/60),sc=s.remaining%60;$('#tm').textContent=String(m).padStart(2,'0')+':'+String(sc).padStart(2,'0');$('#tm').style.color=s.remaining<60?'#ff5468':'';
$('#vc').textContent=s.violations;$('#dot').style.color=s.cls=='CHEAT'?'#ff5468':s.cls=='WARN'?'#f5b93b':'#2ecc71';
s.notes.forEach(n=>{toast('\u26A0 '+n.text+'  ('+s.violations+'/'+s.max+')');speak(n.text)});since=s.nid;
if(!finalWarned&&s.violations>=s.max-2&&s.max>2){finalWarned=true;toast('FINAL WARNING: the exam will auto-submit after '+s.max+' violations','');speak('Final warning. Your exam will be submitted automatically after the next violations.')}
if(window.screen&&screen.isExtended&&Date.now()-lastMM>30000){lastMM=Date.now();vio('multi_monitor');toast('Disconnect the extra monitor')}
if(s.msg.id>lastMsg){lastMsg=s.msg.id;if(s.msg.text){toast('Teacher: '+s.msg.text,'info');speak(s.msg.text)}}}
const vio=t=>{if(inExam&&!ended&&Date.now()-examStart>4000)api('/api/violation',{type:t})};
document.addEventListener('keydown',e=>{if(!inExam||ended)return;const k=(e.key||'').toLowerCase();if(((e.ctrlKey||e.metaKey)&&['c','v','x','a','p','s','u','f','r','n'].includes(k))||k=='f12'||(e.ctrlKey&&e.shiftKey&&['i','j','c'].includes(k))){e.preventDefault();vio('blocked_key')}});
document.addEventListener('keyup',e=>{if(inExam&&!ended&&e.key=='PrintScreen'){try{navigator.clipboard.writeText('')}catch(x){}vio('blocked_key')}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)vio('tab_switch')});
window.addEventListener('blur',()=>{if(inExam)setTimeout(()=>{if(!document.hasFocus())vio('tab_switch')},400)});
document.addEventListener('fullscreenchange',()=>{if(inExam&&!ended&&!document.fullscreenElement){vio('fullscreen_exit');toast('Please stay in fullscreen mode');document.documentElement.requestFullscreen().catch(()=>{})}});
['copy','cut','paste','contextmenu'].forEach(e=>document.addEventListener(e,ev=>{if(inExam){ev.preventDefault();vio(e=='contextmenu'?'right_click':'copy_paste')}}));
/* ---------- ADMIN ---------- */
function adminView(tab){clearInterval(pol);const tabs={live:'\u25CF Live Monitor',students:'Students',results:'Results',exam:'Exam Editor'};
$('#app').innerHTML=`<div class=top><div class=logo>&#128737; Smart<span>Proctor</span> PRO <small style="color:var(--m);font-weight:400">Teacher Console</small></div><div><button class="btn g s" onclick="if(confirm('Stop the proctor system?'))api('/api/admin/shutdown',{}).then(()=>document.body.innerHTML='<p style=padding:40px>System stopped. You can close this tab.</p>')">Stop system</button> <button class="btn g s" onclick=pwModal()>Password</button> <button class="btn g s" onclick=logout()>Logout</button></div></div>
<div style="display:flex"><div class=side>${Object.entries(tabs).map(([k,v])=>`<a class="${k==tab?'on':''}" onclick="adminView('${k}')">${v}</a>`).join('')}</div><div style="flex:1;padding:20px" id=main></div></div>`;
({live:liveTab,students:studTab,results:resTab,exam:examTab})[tab]()}
function gauge(v,c){const C=2*Math.PI*54;return`<svg width=130 height=130 viewBox="0 0 130 130"><circle cx=65 cy=65 r=54 fill=none stroke="#26325a" stroke-width=12 /><circle cx=65 cy=65 r=54 fill=none stroke="${c}" stroke-width=12 stroke-linecap=round stroke-dasharray="${C*v/100} ${C}" transform="rotate(-90 65 65)"/><text x=65 y=72 text-anchor=middle fill="#e8ecf7" font-size=30 font-weight=700>${v}</text></svg>`}
function liveTab(){$('#main').innerHTML=`<div class="grid g2"><div class=card><img class=cam src=/video><div style="display:flex;justify-content:space-between;margin-top:10px;align-items:center"><span id=st class="pill OK">-</span><span id=stu style="color:var(--m)"></span><span id=tmr style="font-weight:700"></span></div></div>
<div class=card style="text-align:center"><small style="color:var(--m)">INTEGRITY SCORE</small><div id=gg></div><div id=gr style="font-weight:700"></div><canvas id=ch width=380 height=90 style="width:100%;margin-top:8px"></canvas></div></div>
<div class="grid g4" style="margin:14px 0" id=kp></div>
<div class="grid g2"><div class=card><b>Live event feed</b><div class=feed id=fd></div></div><div class=card><b>Controls</b><div id=ctl style="margin-top:10px"></div></div></div>`;
const t=async()=>{const d=await api('/api/admin/state');if(!d.ok||!$('#st'))return;live=d;
$('#st').textContent=d.active?d.status:'IDLE';$('#st').className='pill '+(d.active?d.cls:'OK');
$('#stu').textContent=d.active?`${d.student.name} (${d.student.roll}) - answered ${d.answered}/${d.total}`:'No exam running';
$('#tmr').textContent=d.active?String(Math.floor(d.remaining/60)).padStart(2,'0')+':'+String(d.remaining%60).padStart(2,'0'):'';
const c=d.score>=80?'#2ecc71':d.score>=50?'#f5b93b':'#ff5468';$('#gg').innerHTML=gauge(d.active?d.score:100,c);$('#gr').textContent=d.active?d.grade:'-';
const x=$('#ch').getContext('2d');x.clearRect(0,0,380,90);x.strokeStyle=c;x.lineWidth=2;x.beginPath();const h=d.history;h.forEach((v,i)=>{const X=i*380/Math.max(h.length-1,1),Y=88-v*.86;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke();
$('#kp').innerHTML=[['Faces',d.faces],['Violations',d.cheats+'/'+d.max],['Seat distance',d.sensors.dist+' cm'],['Sound',d.sensors.sound],['PIR',d.sensors.pir?'MOTION':'clear'],['Identity match',d.idscore===undefined?'-':d.idscore],['IoT mode',d.mode]].map(a=>`<div class="card kpi"><small>${a[0]}</small><b style="font-size:${a[0]=='IoT mode'?15:24}px">${a[1]}</b></div>`).join('');
$('#fd').innerHTML=d.events.slice(-25).reverse().map(e=>`<div class="${e.event.startsWith('[WARN]')?'s':'h'}">${e.time} &nbsp;<b>${esc(e.event)}</b> ${esc(e.detail)}</div>`).join('')||'<div>No events</div>';
if(!$('#msgi'))$('#ctl').innerHTML=`${d.sim?`<label>IoT simulator (no hardware)</label><div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">${[['talk','Talking'],['away','Seat empty'],['pir','PIR motion'],['btn','Help button']].map(a=>`<button class="btn g s" onclick="api('/api/admin/sim',{k:'${a[0]}'})">${a[1]}</button>`).join('')}</div>`:''}
<label>Message to student</label><div style="display:flex;gap:6px"><input id=msgi placeholder="e.g. Please look at your screen" style="margin:0"><button class="btn s" onclick="api('/api/admin/msg',{text:$('#msgi').value});$('#msgi').value=''">Send</button></div>
<button class="btn r s" style="margin-top:14px" onclick="if(confirm('End this exam now?'))api('/api/admin/end',{})">End exam now</button>`};
t();pol=setInterval(t,1500)}
async function studTab(){const r=await api('/api/admin/students');
$('#main').innerHTML=`<div class=card><h3 style="margin-top:0">Registered students (${r.students.length})</h3>${r.fid.ready?'':`<p style="color:var(--w)">&#9888; ${esc(r.fid.err)}</p>`}
<div style="overflow:auto"><table><tr><th>Photo<th>Name<th>ID<th>Class<th>Contact<th>Face ID<th>Exams<th></tr>
${r.students.map(s=>`<tr><td>${s.photos?`<img src="/api/photo/${s.roll}/1?t=${Date.now()}" style="width:42px;height:42px;object-fit:cover;border-radius:8px">`:'&#128100;'}<td>${esc(s.name)}<td>${esc(s.roll)}<td>${esc(s.branch)} ${esc(s.year)}<td style="font-size:12px">${esc(s.phone)}<br>${esc(s.email)}
<td>${s.enrolled?'<span class="pill OK">Enrolled</span>':'<span class="pill WARN">No photo</span>'}<td>${s.tests}<td><button class="btn s" onclick="manage('${s.roll}')">Manage</button></tr>`).join('')||'<tr><td colspan=8>No students yet. Students register from the login page.</tr>'}</table></div></div>`}
async function manage(roll,msg){const r=await api('/api/admin/students');const s=r.students.find(x=>x.roll==roll);if(!s)return closeM();
const f=(id,l,v,t)=>`<div><label>${l}</label><input id=${id} value="${esc(v||'')}" ${t||''}></div>`;
modal(`<h3 style="margin-top:0">${esc(s.name)} <small style="color:var(--m)">(${esc(s.roll)})</small></h3>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 12px">${f('pn','Name',s.name)}${f('pb','Branch / Class',s.branch)}${f('py','Year',s.year)}${f('pp','Phone',s.phone)}<div style="grid-column:1/3">${f('pe2','Email',s.email)}</div><div style="grid-column:1/3">${f('pw2','Reset student password (optional)','','type=password')}</div></div>
<h4 style="margin:4px 0 8px">Face photos (${s.photos}/3) - used to verify identity</h4><div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">
${[1,2,3].map(n=>`<div style="text-align:center"><img src="/api/photo/${s.roll}/${n}?t=${Date.now()}" onerror="this.style.display='none';this.nextElementSibling.style.display='none'" style="width:84px;height:84px;object-fit:cover;border-radius:10px"><button class="btn r s" style="display:block;margin:4px auto 0;padding:3px 8px" onclick="pdel('${s.roll}',${n})">Delete</button></div>`).join('')}</div>
<input type=file id=pf accept="image/*" style="margin-bottom:8px"><div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn s" onclick="pup('${s.roll}')">Upload photo</button><button class="btn g s" onclick="pcam('${s.roll}')">&#128247; Capture from proctor camera</button></div>
<hr style="border-color:var(--b);margin:14px 0"><label>Your admin password (required for any change)</label><input id=ap type=password>
<div class=err id=pe style="color:${msg?'#2ecc71':'#ff5468'}">${esc(msg||'')}</div><div style="display:flex;gap:8px"><button class=btn onclick="psave('${s.roll}')">Save details</button><button class="btn g" onclick="closeM();studTab()">Close</button></div>`,580)}
async function padm(path,body,roll){body.admin_pw=$('#ap').value;body.roll=roll;if(!body.admin_pw){$('#pe').style.color='#ff5468';$('#pe').textContent='Enter your admin password first';return}
const r=await api(path,body);if(r.ok){const pw=body.admin_pw;await manage(roll,r.msg);$('#ap').value=pw}else{$('#pe').style.color='#ff5468';$('#pe').textContent=r.err}}
function psave(roll){padm('/api/admin/profile',{name:$('#pn').value,branch:$('#pb').value,year:$('#py').value,phone:$('#pp').value,email:$('#pe2').value,newpw:$('#pw2').value},roll)}
function pdel(roll,n){padm('/api/admin/photo_del',{n},roll)}function pcam(roll){padm('/api/admin/photo',{from_camera:true},roll)}
function readImg(file){return new Promise(res=>{const r=new FileReader();r.onload=()=>{const im=new Image();im.onload=()=>{const k=Math.min(1,900/Math.max(im.width,im.height)),c=document.createElement('canvas');c.width=im.width*k;c.height=im.height*k;c.getContext('2d').drawImage(im,0,0,c.width,c.height);res(c.toDataURL('image/jpeg',.9))};im.src=r.result};r.readAsDataURL(file)})}
async function pup(roll){const f=$('#pf').files[0];if(!f){$('#pe').style.color='#ff5468';return $('#pe').textContent='Choose a photo file first'}padm('/api/admin/photo',{image:await readImg(f)},roll)}
function pwModal(){modal(`<h3 style="margin-top:0">Change admin password</h3><label>Current password</label><input id=o type=password><label>New password</label><input id=nw type=password><div class=err id=pe></div>
<button class=btn onclick="api('/api/admin/passwd',{old:$('#o').value,new:$('#nw').value}).then(r=>{$('#pe').style.color=r.ok?'#2ecc71':'#ff5468';$('#pe').textContent=r.ok?r.msg:r.err})">Change</button> <button class="btn g" onclick=closeM()>Close</button>`)}
async function resTab(){const r=await api('/api/admin/results');$('#main').innerHTML=`<div class=card><div style="display:flex;justify-content:space-between"><h3 style="margin-top:0">Results (${r.results.length})</h3><a class="btn s" style="text-decoration:none" href=/api/admin/export>Export CSV</a></div><div style="overflow:auto"><table><tr><th>Date<th>Student<th>Score<th>Integrity<th>Violations<th>Identity<th>Verdict<th>Ended<th>Report</tr>
${r.results.map(x=>`<tr><td>${x.date}<td>${esc(x.name)} (${esc(x.roll)})<td>${x.marks!==undefined?x.marks+'/'+x.max_marks+' ('+x.percent+'%)':x.correct+'/'+x.total}<td>${x.integrity}<td>${x.violations}<td>${x.id_checks?x.id_ok+'/'+x.id_checks:'-'}<td><span class="pill ${x.verdict=='TRUSTED'?'OK':x.verdict=='SUSPICIOUS'?'CHEAT':'WARN'}">${x.verdict}</span><td>${esc(x.reason)}<td><a href="/report/${x.sid}" target=_blank>Open</a></tr>`).join('')||'<tr><td colspan=9>No results yet</tr>'}</table></div></div>`}
async function examTab(id){const pl=await api('/api/admin/papers');id=id||pl.active;const r=await api('/api/admin/exam?id='+id);
$('#main').innerHTML=`<div class=card><div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px"><b>Paper:</b><select id=ps style="width:auto;margin:0" onchange="examTab(this.value)">${pl.papers.map(p=>`<option value="${p.id}" ${p.id==id?'selected':''}>${esc(p.title)} (${p.n} Q, ${p.minutes} min)${p.id==pl.active?' - ACTIVE':''}</option>`).join('')}</select>
<button class="btn s" onclick="paperAct('select','${id}')" ${id==pl.active?'disabled':''}>Make active</button><button class="btn g s" onclick="newPaper()">+ New paper</button><button class="btn r s" onclick="if(confirm('Delete this paper?'))paperAct('del','${id}')">Delete</button></div>
<p style="color:var(--m);font-size:13px;line-height:1.7">Question types: <code>single</code> (MCQ, answer = option index from 0), <code>multi</code> (answer = list like [0,2]), <code>tf</code> (options True/False), <code>num</code> (answer = number, optional <code>tol</code>), <code>text</code> (answer = list of accepted words). Each question can have <code>section</code>, <code>marks</code>, <code>code</code>. Paper settings: <code>minutes</code>, <code>negative</code> (e.g. 0.25), <code>shuffle</code>, <code>access_code</code>, <code>max_violations</code>, <code>attempts</code>, <code>require_identity</code>, <code>voice_check</code>.</p>
<textarea id=ej rows=20 style="font-family:Consolas,monospace;font-size:13px">${esc(JSON.stringify(r.exam,null,2))}</textarea><div class=err id=ee></div><button class=btn onclick="saveExam('${id}')">Save paper</button></div>`}
async function paperAct(act,id,extra){const r=await api('/api/admin/paper_'+act,Object.assign({id},extra||{}));if(!r.ok)return alert(r.err);examTab(act=='del'?undefined:(r.id||id))}
function newPaper(){const id=prompt('New paper id (letters/numbers, e.g. test2):');if(!id)return;const t=prompt('Paper title:',id);paperAct('new',id.toLowerCase(),{title:t})}
async function saveExam(id){let e;try{e=JSON.parse($('#ej').value)}catch(x){return $('#ee').textContent='Invalid JSON: '+x.message}const r=await api('/api/admin/paper_save',{id,exam:e});$('#ee').style.color=r.ok?'#2ecc71':'#ff5468';$('#ee').textContent=r.ok?'Saved!':r.err}
init();
</script></body></html>"""
