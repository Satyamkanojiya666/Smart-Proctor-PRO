// ESP32 Desk Node for AI Smart Exam Proctoring
// Sends JSON sensor lines to PC over USB serial, receives STATE/BUZZ/SCORE commands.
//
// WIRING (ESP32 DevKit)
//  HC-SR04  TRIG -> GPIO 5    ECHO -> GPIO 18 (use 5V->3.3V divider: 1k + 2k)   VCC 5V, GND
//  PIR HC-SR501 OUT -> GPIO 27  VCC 5V, GND
//  Sound module (KY-037/MAX4466) AO -> GPIO 34 (ADC)  VCC 3.3V, GND
//  Push button (call invigilator) -> GPIO 14 and GND (INPUT_PULLUP)
//  RGB LED (common cathode) R -> GPIO 25, G -> GPIO 26, B -> GPIO 33 via 220 ohm each
//  Active buzzer + -> GPIO 23, - -> GND

#define TRIG 5
#define ECHO 18
#define PIR  27
#define MIC  34
#define BTN  14
#define LR   25
#define LG   26
#define LB   33
#define BUZ  23

String state = "OK";
unsigned long lastSend = 0, buzzUntil = 0, lastBlink = 0;
bool blinkOn = false;

void setLed(int r, int g, int b) {
  digitalWrite(LR, r); digitalWrite(LG, g); digitalWrite(LB, b);
}

int readDistanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long d = pulseIn(ECHO, HIGH, 30000);
  if (d == 0) return 300;              // no echo = far / empty
  return (int)(d * 0.0343 / 2);
}

int readSoundPeak() {                  // peak-to-peak over 50 ms
  int mn = 4095, mx = 0;
  unsigned long t = millis();
  while (millis() - t < 50) {
    int v = analogRead(MIC);
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  return (mx - mn);
}

void handleCommand(String c) {
  c.trim();
  if (c.startsWith("STATE:")) state = c.substring(6);
  else if (c.startsWith("BUZZ:")) buzzUntil = millis() + c.substring(5).toInt();
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT); pinMode(PIR, INPUT);
  pinMode(BTN, INPUT_PULLUP);
  pinMode(LR, OUTPUT); pinMode(LG, OUTPUT); pinMode(LB, OUTPUT); pinMode(BUZ, OUTPUT);
  setLed(0, 0, 1);                     // blue = waiting for PC
}

void loop() {
  while (Serial.available()) {
    static String buf;
    char ch = Serial.read();
    if (ch == '\n') { handleCommand(buf); buf = ""; } else buf += ch;
  }

  if (millis() - lastSend >= 200) {
    lastSend = millis();
    Serial.printf("{\"dist\":%d,\"pir\":%d,\"sound\":%d,\"btn\":%d}\n",
                  readDistanceCm(), digitalRead(PIR), readSoundPeak(), !digitalRead(BTN));
  }

  if (state == "OK")        setLed(0, 1, 0);
  else if (state == "WARN") setLed(1, 1, 0);
  else {                                // CHEAT: fast red blink
    if (millis() - lastBlink > 250) { lastBlink = millis(); blinkOn = !blinkOn; }
    setLed(blinkOn, 0, 0);
  }
  digitalWrite(BUZ, millis() < buzzUntil ? HIGH : LOW);
}
