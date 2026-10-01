"""
IoT layer for the AI Exam Proctoring System.

- ESP32 desk node talks over USB serial (JSON lines).  No hardware? -> simulator mode.
- Sensor fusion: camera + ultrasonic + PIR + sound + button.
- Live Integrity Score (0-100) and a web dashboard (stdlib only, no Flask).

Serial protocol
  ESP32 -> PC : {"dist":52,"pir":0,"sound":310,"btn":0}\n     (about 5 Hz)
  PC -> ESP32 : STATE:OK | STATE:WARN | STATE:CHEAT | BUZZ:<ms> | SCORE:<0-100>\n
"""
import json, os, random, threading, time, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import serial, serial.tools.list_ports
except ImportError:
    serial = None

# ---- tunables (cm / ADC units / seconds) ------------------------------------
SEAT_MAX_CM      = 90     # ultrasonic reading above this = seat empty
SEAT_LEAVE_SEC   = 3.0
SOUND_THRESHOLD  = 1800   # 0-4095 ADC (ESP32 mic module)
SOUND_SUSTAIN    = 1.5
COVER_SEC        = 4.0    # camera sees no face but seat occupied
SPOOF_SEC        = 6.0    # camera sees face but seat empty


class Fusion:
    """Cross-checks camera result with hardware sensors. Returns list of (level, event, detail)."""
    def __init__(self):
        self.t = collections.defaultdict(lambda: None)
        self.last_btn = 0

    def _held(self, key, cond, sec):
        if not cond:
            self.t[key] = None
            return False
        if self.t[key] is None:
            self.t[key] = time.time()
        return time.time() - self.t[key] >= sec

    def check(self, s, face_count):
        out = []
        seat_empty = s["dist"] > SEAT_MAX_CM
        if self._held("seat", seat_empty and face_count == 0, SEAT_LEAVE_SEC):
            out.append(("hard", "seat_left", f"ultrasonic={s['dist']}cm, no face"))
        if self._held("cover", (not seat_empty) and face_count == 0, COVER_SEC):
            out.append(("hard", "camera_tampered",
                        f"seat occupied ({s['dist']}cm) but camera sees no face"))
        if self._held("spoof", seat_empty and face_count >= 1, SPOOF_SEC):
            out.append(("hard", "possible_spoof",
                        f"face on camera but seat empty ({s['dist']}cm) - photo/video?"))
        if self._held("sound", s["sound"] > SOUND_THRESHOLD, SOUND_SUSTAIN):
            out.append(("hard", "talking_detected", f"sound={s['sound']}"))
        if s["pir"] and face_count <= 1 and not seat_empty:
            if self._held("pir", True, 2.0):
                out.append(("soft", "person_near_desk", "PIR motion beside desk"))
        else:
            self.t["pir"] = None
        if s["btn"] and not self.last_btn:
            out.append(("soft", "help_request", "candidate pressed call button"))
        self.last_btn = s["btn"]
        return out


class IntegrityScore:
    """100 = perfectly clean. Hard events cost 6, soft 1.5, slow recovery."""
    def __init__(self):
        self.v = 100.0
        self.t = time.time()
        self.history = collections.deque(maxlen=300)

    def hit(self, level):
        self.v = max(0.0, self.v - (6.0 if level == "hard" else 1.5))

    def tick(self):
        now = time.time()
        self.v = min(100.0, self.v + 0.15 * (now - self.t))
        self.t = now
        self.history.append(round(self.v, 1))
        return self.v

    @property
    def grade(self):
        return ("Trusted" if self.v >= 80 else "Watch" if self.v >= 50 else "Suspicious")


class IoTNode:
    def __init__(self, port=None, baud=115200, simulate=False):
        self.latest = {"dist": 55, "pir": 0, "sound": 300, "btn": 0}
        self.connected = False
        self.mode = "simulator"
        self._ser = None
        self._lock = threading.Lock()
        self._stop = False
        self.sim_keys = {"talk": False, "away": False, "pir": False}
        if not simulate and serial is not None:
            port = port or self._autodetect()
            if port:
                try:
                    self._ser = serial.Serial(port, baud, timeout=0.5)
                    time.sleep(1.8)  # ESP32 resets on open
                    self.mode, self.connected = f"hardware:{port}", True
                except Exception as e:
                    print(f"  [IoT] could not open {port}: {e} -> simulator")
        if self._ser is None:
            self.connected = True
        threading.Thread(target=self._reader, daemon=True).start()

    @staticmethod
    def _autodetect():
        for p in serial.tools.list_ports.comports():
            d = (p.description or "").lower()
            if any(k in d for k in ("cp210", "ch340", "ch910", "usb serial", "esp32", "uart")):
                return p.device
        return None

    def _reader(self):
        while not self._stop:
            if self._ser is None:
                self._simulate(); time.sleep(0.2); continue
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    with self._lock:
                        self.latest.update({k: d[k] for k in ("dist", "pir", "sound", "btn") if k in d})
            except Exception:
                time.sleep(0.05)

    def _simulate(self):
        k = self.sim_keys
        with self._lock:
            self.latest["dist"] = 140 if k["away"] else 55 + random.randint(-4, 4)
            self.latest["sound"] = 2600 if k["talk"] else 280 + random.randint(0, 120)
            self.latest["pir"] = 1 if k["pir"] else 0

    def sim_toggle(self, name):
        if name in self.sim_keys:
            self.sim_keys[name] = not self.sim_keys[name]
            return self.sim_keys[name]

    def sim_button(self):
        with self._lock:
            self.latest["btn"] = 1
        threading.Timer(0.6, lambda: self.latest.__setitem__("btn", 0)).start()

    def read(self):
        with self._lock:
            return dict(self.latest)

    def send(self, msg):
        if self._ser:
            try: self._ser.write((msg + "\n").encode())
            except Exception: pass

    def close(self):
        self._stop = True
        self.send("STATE:OK")
        if self._ser: self._ser.close()


# ---- dashboard ---------------------------------------------------------------
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Smart Proctor Dashboard</title>
<style>
body{font-family:system-ui;background:#0f1420;color:#e8ecf4;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 12px}.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.c{background:#1a2133;border-radius:12px;padding:12px}.c b{display:block;font-size:26px;margin-top:4px}
#st{font-size:22px;font-weight:700}.OK{color:#3ddc84}.WARN{color:#ffc247}.CHEAT{color:#ff5468}
canvas{width:100%;height:120px;background:#1a2133;border-radius:12px;margin:10px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:5px;border-bottom:1px solid #263049}
</style>
<h1>Smart Exam Proctor - Live</h1>
<div id=stu style="margin:-6px 0 10px;color:#9fb0d0"></div>
<div class=g>
<div class=c>Status<b id=st>-</b></div><div class=c>Integrity<b id=sc>-</b><small id=gr></small></div>
<div class=c>Seat distance<b id=d>-</b></div><div class=c>Sound<b id=s>-</b></div>
<div class=c>PIR<b id=p>-</b></div><div class=c>Faces<b id=f>-</b></div>
<div class=c>Cheat events<b id=ce>-</b></div><div class=c>IoT mode<b id=m style="font-size:14px">-</b></div></div>
<canvas id=cv width=600 height=120></canvas>
<table id=ev></table>
<script>
async function u(){try{const r=await (await fetch('/api/state')).json();
st.textContent=r.status;st.className=r.cls;sc.textContent=r.score;gr.textContent=r.grade;
d.textContent=r.sensors.dist+' cm';s.textContent=r.sensors.sound;p.textContent=r.sensors.pir?'MOTION':'clear';
stu.textContent=(r.student||'')+'   '+(r.left||'');f.textContent=r.faces;ce.textContent=r.cheats;m.textContent=r.mode;
const x=cv.getContext('2d');x.clearRect(0,0,600,120);x.strokeStyle='#3ddc84';x.beginPath();
r.history.forEach((v,i)=>{const X=i*600/Math.max(r.history.length-1,1),Y=120-v*1.2;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke();
ev.innerHTML=r.events.slice(-12).reverse().map(e=>`<tr><td>${e.time}</td><td>${e.event}</td><td>${e.detail}</td></tr>`).join('')}catch(e){}}
setInterval(u,1000);u();
</script>"""


class Dashboard:
    def __init__(self, port=8000):
        self.state = {"status": "-", "cls": "OK", "score": 100, "grade": "Trusted",
                      "sensors": {"dist": 0, "sound": 0, "pir": 0}, "faces": 0,
                      "cheats": 0, "mode": "-", "history": [], "events": []}
        st = self.state

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path.startswith("/api/state"):
                    body, ct = json.dumps(st).encode(), "application/json"
                else:
                    body, ct = PAGE.encode(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0", port), H)
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            self.port = port
        except OSError:
            self.srv = None; self.port = None

    def update(self, **kw):
        self.state.update(kw)

    def close(self):
        if self.srv: self.srv.shutdown()
