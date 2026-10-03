# IoT Setup Guide

## Hardware (approx ₹1200-1500)
ESP32 DevKit, HC-SR04 ultrasonic, HC-SR501 PIR, KY-037 / MAX4466 sound module,
push button, RGB LED (common cathode) + 3x220Ω, active buzzer, 1k+2k resistors (echo divider), breadboard, USB cable.
Wiring is written at the top of `firmware/esp32_proctor_node/esp32_proctor_node.ino`.

## Steps
1. Arduino IDE -> Board: "ESP32 Dev Module" -> open the .ino -> Upload.
2. `pip install -r requirements.txt`
3. Run with hardware:  `python smart_proctor_pro.py`  (port auto-detect)  or  `--port COM3`
4. Run WITHOUT hardware (demo/viva): `python smart_proctor_pro.py --simulate`
   keys: T=talking, A=student away, P=PIR motion, B=help button
5. Dashboard: http://localhost:8000 (phone: http://<PC-IP>:8000)

## Unique features
- Sensor fusion: camera_tampered (camera covered but seat occupied), possible_spoof (photo/video on camera, seat empty), seat_left, talking_detected, person_near_desk, help_request
- Live Integrity Score 0-100 (LED + dashboard + report)
- Desk LED/buzzer feedback: green / yellow / blinking red
- Auto evidence snapshot in `evidence/` on every hard event
- Web dashboard with live graph and event feed
