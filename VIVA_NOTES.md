# Viva Notes - Smart Proctor PRO

## Ek line me project
"Ye ek online exam platform hai jisme **Camera AI + IoT sensors + browser anti-cheat** ko jodkar cheating pakdi jati hai, aur student ki **identity face + voice se verify** hoti hai."

## Flow
Student register/login -> face verify + voice code -> exam (timer, palette) -> live proctoring -> auto report.
Teacher: live monitor, message bhejna, results, CSV, photo enrolment (password-locked), exam editor.

## Technology
| Kaam | Kya use hua |
|---|---|
| Face, gaze, head pose | MediaPipe FaceLandmarker (478 landmarks), EAR se blink |
| Phone / laptop detection | YOLOv4-tiny (OpenCV DNN) |
| Identity match | OpenCV YuNet (detect) + SFace (recognise), cosine similarity |
| IoT | ESP32 + ultrasonic + PIR + sound + button + LED/buzzer (ya simulator) |
| Web app | Python http.server, HTML/JS, MJPEG live video |
| Passwords | PBKDF2-SHA256 + salt (plain text me save nahi hote) |

## Unique features (original project me nahi the)
1. **camera_tampered**: camera dhaka hai par ultrasonic batata hai banda seat pe hai.
2. **possible_spoof**: camera me chehra hai par seat khali -> photo/video ka shak.
3. **Identity verification**: exam se pehle + har 5 sec exam ke dauran.
4. **Voice liveness**: random 4-digit code bolna (replay attack se bachata hai).
5. **Pattern AI**: ek taraf baar-baar dekhna, ya kam time me kai violations.
6. **Integrity Score 0-100** + verdict (Trusted / Needs Review / Suspicious).
7. **Browser anti-cheat**: tab switch, fullscreen exit, copy-paste, right-click.
8. **Auto evidence photos** + HTML report (timeline graph, kisne pakda: Camera/IoT/Device/Identity).

## Naye advanced features (v4)
- **Device detection (YOLO)**: phone, laptop/tablet, remote = hard violation; mouse, keyboard, book/notes, watch = soft warning; **extra person** (body detection, jab chehra na dikhe tab bhi). Ek device tabhi confirm hota hai jab 5 me se 3 baar dikhe (false alarm kam).
- **Mic monitoring (browser)**: shuruaat me 3 sec ka calibration, phir lagatar shor = `mic_noise`, 6 sec me zyada bolna = `mic_talking`. IoT sound sensor ke bina bhi kaam karta hai.
- **Voice warnings**: har violation par bol kar warning + toast, aur "Final warning" jab limit ke paas ho.
- **Exam paper**: 5 question types (MCQ, multi-select, true/false, number, short text), sections, code snippets, marks, **negative marking**, **question/option shuffle**, access code, section-wise result.
- **Paper library**: Full Demo Paper (24 Q, 20 min) aur Quick Demo Paper (6 Q, 3 min); teacher naye paper bana/edit/active kar sakta hai.
- **Browser anti-cheat+**: multi-monitor detection, blocked shortcuts (Ctrl+C/V, F12, PrintScreen), tab switch, fullscreen exit.

## Possible sawal aur jawab
- **Sirf camera kyu kaafi nahi?** Camera dhaka ya photo dikhaye to AI ko pata nahi chalta; IoT sensor physical presence batata hai.
- **Face match kaise hota hai?** Chehre se 128-number ka embedding banta hai; live aur enrolled embedding ka cosine similarity threshold (0.363) se compare hota hai.
- **Voice check voiceprint hai?** Nahi. Ye liveness check hai: random code bolna padta hai, isliye recorded video kaam nahi karta. Voiceprint future work hai.
- **False alarm kaise kam kiya?** Cooldown, smoothing, 3 lagatar mismatch par hi identity flag, sirf seedhe chehre par check.
- **Privacy?** Photos aur data local `data/` folder me, cloud pe nahi. Student ko exam se pehle consent (checkbox) milta hai.
- **Hardware ke bina?** Simulator mode: teacher console ke buttons se sensors ki nakal hoti hai.

## Limitations (honestly batana)
- Ek camera = ek time pe ek student. Bahut students ke liye har PC pe client aur central server chahiye (future work).
- Bahut kam roshni me face match kamzor ho sakta hai.
- Browser voice recognition ke liye Chrome/Edge + internet chahiye.

## Future work
Multi-student server, voiceprint, mobile second-camera, cloud dashboard, Telegram alerts.
