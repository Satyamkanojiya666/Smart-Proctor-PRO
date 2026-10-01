"""
extras.py - Advanced add-on features for Smart Proctor IoT

1. Student login (name + roll no) -> shown on dashboard + report
2. Voice warnings (pyttsx3, event-specific messages, optional)
3. PatternWatch  - behaviour-pattern AI: burst of violations, repeated
                   glances in one direction (possible notes / helper)
4. Timed exam    - --minutes N countdown, auto-end
5. Rich HTML report - integrity timeline graph, camera-vs-IoT-vs-device
                   breakdown, embedded evidence photos, final verdict
"""
import os, io, time, html, base64, queue, threading, collections, webbrowser
from datetime import datetime

# --------------------------------------------------------------------------
# event categories (used in report: shows what ONLY IoT fusion could catch)
# --------------------------------------------------------------------------
IOT_EVENTS = {"seat_left", "camera_tampered", "possible_spoof",
              "talking_detected", "person_near_desk", "help_request"}

BROWSER_EVENTS = {"tab_switch", "fullscreen_exit", "copy_paste", "right_click", "multi_monitor", "blocked_key"}
AUDIO_EVENTS = {"mic_noise", "mic_talking"}

def category(event):
    e = event.replace("[WARN]", "").strip()
    if e in AUDIO_EVENTS:
        return "Mic (audio)"
    if e in BROWSER_EVENTS:
        return "Browser"
    if e in IOT_EVENTS:
        return "IoT sensor"
    if e.startswith("device_") or "device" in e:
        return "Device (YOLO)"
    if e == "identity_mismatch":
        return "Identity AI"
    if e == "suspicious_pattern" or e == "repeated_glance":
        return "Pattern AI"
    return "Camera AI"


# --------------------------------------------------------------------------
# 1. student login
# --------------------------------------------------------------------------
def student_login(skip=False):
    if skip:
        return {"name": "Demo Student", "roll": "000"}
    print("\n" + "=" * 50)
    print("   SMART PROCTOR - STUDENT LOGIN")
    print("=" * 50)
    name = input("  Student name : ").strip() or "Unknown"
    roll = input("  Roll number  : ").strip() or "NA"
    print(f"  Welcome {name} ({roll}). Exam starting...\n")
    return {"name": name, "roll": roll}


# --------------------------------------------------------------------------
# 2. voice warnings
# --------------------------------------------------------------------------
VOICE_MSG = {
    "eye_gaze_left":   "Please look at your screen.",
    "eye_gaze_right":  "Please look at your screen.",
    "head_left":       "Please face the screen.",
    "head_right":      "Please face the screen.",
    "head_tilt_up":    "Please keep your head straight.",
    "face_tilt_down":  "Please do not look down.",
    "no_face":         "Face not visible. Please sit in front of the camera.",
    "multiple_faces":  "Another person detected. This has been recorded.",
    "talking_detected": "Talking detected. Please stay silent.",
    "seat_left":       "You have left your seat. This has been recorded.",
    "camera_tampered": "Camera appears blocked. Please uncover it.",
    "possible_spoof":  "Live presence not verified. Please sit at your desk.",
    "suspicious_pattern": "Suspicious behaviour pattern detected. Teacher notified.",
    "repeated_glance": "Repeated looking away detected.",
    "identity_mismatch": "Identity check failed. Please show your face clearly.",
    "mic_talking":     "Talking detected. Please stay silent.",
    "multi_monitor":   "Multiple screens detected. Disconnect the extra monitor.",
    "device_cell_phone": "Mobile phone detected. Put it away immediately.",
    "device_laptop":   "Another laptop or tablet detected. Remove it.",
    "device_remote":   "Electronic device detected. Remove it.",
    "device_extra_person": "Another person is in the room. This has been recorded.",
    "tab_switch":      "You left the exam window. This has been recorded.",
    "fullscreen_exit": "Please stay in fullscreen mode.",
}

class Voice:
    def __init__(self, enabled=True, gap=6.0):
        self.enabled, self.gap = enabled, gap
        self._last, self._q = 0.0, queue.Queue(maxsize=2)
        self.ok = False
        if enabled:
            try:
                import pyttsx3  # noqa
                self.ok = True
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception:
                print("  [Voice] pyttsx3 not installed -> voice off (pip install pyttsx3)")

    def _worker(self):
        import pyttsx3
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", 165)
        except Exception:
            self.ok = False
            return
        while True:
            text = self._q.get()
            try:
                eng.say(text); eng.runAndWait()
            except Exception:
                pass

    def say_event(self, event):
        if not (self.enabled and self.ok):
            return
        msg = VOICE_MSG.get(event)
        if not msg or time.time() - self._last < self.gap:
            return
        self._last = time.time()
        try: self._q.put_nowait(msg)
        except queue.Full: pass


# --------------------------------------------------------------------------
# 3. behaviour-pattern AI
# --------------------------------------------------------------------------
class PatternWatch:
    """Looks at the *history* of events, not single frames."""
    BURST_N, BURST_SEC       = 4, 45     # 4 hard events in 45 s
    GLANCE_N, GLANCE_SEC     = 5, 60     # 5 same-side glances in 60 s

    def __init__(self):
        self.hard = collections.deque()
        self.side = {"left": collections.deque(), "right": collections.deque()}
        self._cool = {}

    def add(self, event):
        if event in ("suspicious_pattern", "repeated_glance"):
            return
        now = time.time()
        self.hard.append(now)
        if event in ("eye_gaze_left", "head_left"):   self.side["left"].append(now)
        if event in ("eye_gaze_right", "head_right"): self.side["right"].append(now)

    def _ok(self, key, sec=30):
        now = time.time()
        if now - self._cool.get(key, 0) >= sec:
            self._cool[key] = now
            return True
        return False

    def check(self):
        now, out = time.time(), []
        while self.hard and now - self.hard[0] > self.BURST_SEC: self.hard.popleft()
        for s, dq in self.side.items():
            while dq and now - dq[0] > self.GLANCE_SEC: dq.popleft()
            if len(dq) >= self.GLANCE_N and self._ok("g_" + s):
                out.append(("repeated_glance",
                            f"looked {s.upper()} {len(dq)}x in {self.GLANCE_SEC}s - notes/helper on {s}?"))
                dq.clear()
        if len(self.hard) >= self.BURST_N and self._ok("burst"):
            out.append(("suspicious_pattern",
                        f"{len(self.hard)} violations within {self.BURST_SEC}s - burst behaviour"))
            self.hard.clear()
        return out


class DeviceTracker:
    """Confirms a device only if seen in >=3 of the last 5 detection cycles (fewer false alarms)."""
    def __init__(self, need=3, window=5, cooldown=8.0):
        self.need, self.window, self.cool = need, window, cooldown
        self.h, self.last = {}, {}
    def update(self, present):
        for cid in set(self.h) | set(present):
            self.h.setdefault(cid, collections.deque(maxlen=self.window)).append(cid in present)
    def confirmed(self, cid):
        d = self.h.get(cid)
        return bool(d) and sum(d) >= self.need
    def ready(self, cid):
        now = time.time()
        if now - self.last.get(cid, 0) >= self.cool:
            self.last[cid] = now
            return True
        return False


# --------------------------------------------------------------------------
# 4. timeline sampler + exam timer
# --------------------------------------------------------------------------
class Timeline:
    def __init__(self):
        self.pts, self._t = [], 0.0
    def add(self, t, score):
        if t - self._t >= 1.0:
            self._t = t
            self.pts.append((round(t, 1), round(score, 1)))

def remaining_text(limit_sec, elapsed):
    if not limit_sec:
        return ""
    r = max(0, int(limit_sec - elapsed))
    return f"Time left {r//60:02d}:{r%60:02d}"


# --------------------------------------------------------------------------
# 5. HTML report
# --------------------------------------------------------------------------
def _verdict(score, hard_events):
    if score >= 80 and hard_events <= 2:
        return "TRUSTED", "#2e9e5b", "No significant violations. No action needed."
    if score >= 50:
        return "NEEDS REVIEW", "#d9930d", "Some violations recorded. Teacher should review evidence."
    return "SUSPICIOUS", "#d64545", "Multiple serious violations. Manual investigation recommended."

def _timeline_svg(pts, total):
    W, H = 700, 180
    if len(pts) < 2:
        return "<p>Not enough data for timeline.</p>"
    mx = max(total, pts[-1][0], 1)
    path = " ".join(f"{'M' if i == 0 else 'L'}{p[0]/mx*W:.1f},{H-p[1]*H/100:.1f}"
                    for i, p in enumerate(pts))
    grid = "".join(f"<line x1='0' x2='{W}' y1='{H-v*H/100}' y2='{H-v*H/100}' stroke='#ddd'/>"
                   f"<text x='2' y='{H-v*H/100-2}' font-size='10' fill='#888'>{v}</text>"
                   for v in (25, 50, 75, 100))
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='background:#fafafa;border:1px solid #ddd'>"
            f"{grid}<path d='{path}' fill='none' stroke='#2e7de0' stroke-width='2'/></svg>")

def _bars(counts):
    if not counts:
        return "<p>No violations.</p>"
    mx = max(counts.values())
    colors = {"Camera AI": "#2e7de0", "IoT sensor": "#e0782e",
              "Device (YOLO)": "#8a2ee0", "Pattern AI": "#d64545", "Browser": "#2e9e5b", "Identity AI": "#c2185b", "Mic (audio)": "#00897b"}
    return "".join(
        f"<div style='margin:4px 0'><span style='display:inline-block;width:130px'>{html.escape(k)}</span>"
        f"<span style='display:inline-block;height:16px;width:{v/mx*320:.0f}px;background:{colors.get(k,'#888')};"
        f"vertical-align:middle'></span> <b>{v}</b></div>" for k, v in counts.items())

def _evidence(sid):
    try:
        import cv2
    except Exception:
        return ""
    out = ""
    if not os.path.isdir("evidence"):
        return "<p>No evidence photos.</p>"
    for f in sorted(os.listdir("evidence")):
        if not (f.startswith(sid) and f.lower().endswith((".jpg", ".png"))):
            continue
        img = cv2.imread(os.path.join("evidence", f))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (320, int(320 * h / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        label = f[len(sid) + 1:].rsplit(".", 1)[0]
        out += (f"<figure style='display:inline-block;margin:6px;text-align:center'>"
                f"<img src='data:image/jpeg;base64,{base64.b64encode(buf).decode()}'>"
                f"<figcaption style='font-size:12px'>{html.escape(label)}</figcaption></figure>")
    return out or "<p>No evidence photos.</p>"

def make_html_report(sid, student, events, stats, score, grade, duration_sec,
                     cheat_count, timeline_pts, iot_mode, path=None, open_browser=True, identity=None):
    path = path or f"logs/report_{sid}.html"
    hard = [e for e in events if not str(e["event"]).startswith("[WARN]")]
    cats = collections.Counter(category(e["event"]) for e in hard)
    verdict, vcol, vtxt = _verdict(score, len(hard))
    iot_only = [e for e in hard if e["event"] in ("camera_tampered", "possible_spoof")]
    rows = "".join(
        f"<tr><td>{e['time']}</td><td>{html.escape(str(e['event']))}</td>"
        f"<td>{category(e['event'])}</td><td>{html.escape(str(e['detail']))}</td></tr>"
        for e in events)
    dur = f"{int(duration_sec)//60}m {int(duration_sec)%60}s"
    idsec = ""
    if identity:
        ph = ""
        try:
            import cv2
            im = cv2.imread(identity["photo"]) if identity.get("photo") else None
            if im is not None:
                im = cv2.resize(im, (110, int(110 * im.shape[0] / im.shape[1])))
                ph = ("<img style='float:left;margin-right:14px;border-radius:8px' src='data:image/jpeg;base64,"
                      + base64.b64encode(cv2.imencode('.jpg', im)[1]).decode() + "'>")
        except Exception:
            pass
        idsec = (f"<h2>Identity Verification</h2><div style='overflow:hidden'>{ph}"
                 f"<p><b>Pre-exam face check:</b> {'PASSED' if identity.get('pre_face') else 'not done'} &nbsp; "
                 f"<b>Voice code:</b> {html.escape(str(identity.get('pre_voice', '-')))}<br>"
                 f"<b>Continuous checks during exam:</b> {identity.get('ok', 0)} matched of {identity.get('checks', 0)}"
                 f" &nbsp; <b>Lowest similarity:</b> {identity.get('min') if identity.get('min') is not None else '-'}</p></div>")
    page = f"""<!doctype html><meta charset=utf-8><title>Proctor Report {sid}</title>
<style>body{{font-family:Segoe UI,Arial;margin:30px;max-width:900px}}
h1{{margin-bottom:0}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px;text-align:left}}th{{background:#f0f3f8}}
.box{{display:inline-block;border:1px solid #ddd;border-radius:10px;padding:10px 18px;margin:4px;text-align:center}}
.box b{{display:block;font-size:24px}}@media print{{body{{margin:10px}}}}</style>
<h1>AI Exam Proctoring Report</h1><small>Camera AI + IoT Sensor Fusion &nbsp;|&nbsp; Session {sid}</small>
<hr>
<p><b>Student:</b> {html.escape(student['name'])} &nbsp; <b>Roll No:</b> {html.escape(student['roll'])}
&nbsp; <b>Date:</b> {datetime.now():%d-%m-%Y %H:%M} &nbsp; <b>Duration:</b> {dur}
&nbsp; <b>IoT mode:</b> {html.escape(str(iot_mode))}</p>
<div>
<div class=box>Integrity Score<b>{score}/100</b>{html.escape(str(grade))}</div>
<div class=box>Violations<b>{cheat_count}</b>hard events</div>
<div class=box>IoT-only catches<b>{len(iot_only)}</b>tamper / spoof</div>
<div class=box style='border-color:{vcol};color:{vcol}'>Verdict<b>{verdict}</b></div></div>
<p>{vtxt}</p>
{idsec}<h2>Integrity Score Timeline</h2>{_timeline_svg(timeline_pts, duration_sec)}
<h2>Who caught it? (Camera vs IoT vs Device vs Pattern)</h2>{_bars(cats)}
<h2>Event Log ({len(events)})</h2>
<table><tr><th>Time</th><th>Event</th><th>Detected by</th><th>Detail</th></tr>{rows}</table>
<h2>Evidence Photos</h2>{_evidence(sid)}
<hr><small>Generated automatically by Smart Proctor IoT. Print -> Save as PDF to submit.</small>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    if open_browser:
        try: webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))
        except Exception: pass
    return path
