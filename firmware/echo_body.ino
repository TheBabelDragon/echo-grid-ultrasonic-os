/**
 * Echo Body — Ultrasonic physical body with hardware-ready sensing
 *
 * Sensing path is designed for eventual real hardware:
 *   - Electret / MEMS microphone
 *   - Ultrasonic receiver transducer
 *   - Or any analog envelope / amplitude signal into an ADC pin
 *
 * Current implementation uses ESP32 ADC (12-bit).
 * Swap the AcousticSensor internals later for I2S or external ADC
 * without changing the Field Body Protocol or host side.
 */

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Configuration — change these to match your wiring
// ---------------------------------------------------------------------------
#ifndef ECHO_BODY_NODE_ID
#define ECHO_BODY_NODE_ID "echo_us_001"
#endif

const int NUM_EMITTERS = 4;
const int TRANSDUCER_PINS[NUM_EMITTERS] = {25, 26, 27, 14};  // PWM outputs
const int PWM_CHANNELS[NUM_EMITTERS]    = {0, 1, 2, 3};
const int PWM_RESOLUTION = 8;
const int BASE_FREQ_HZ   = 40000;

// Acoustic sensor (microphone / ultrasonic receiver)
const int SENSOR_PIN          = 34;   // ADC1 channel (input-only pin on ESP32)
const int SENSOR_SAMPLES      = 48;   // how many ADC reads to average
const float SENSOR_VREF       = 3.3f;
const int SENSOR_BITS         = 12;   // ESP32 ADC default
const float SENSOR_MAX_COUNTS = (float)((1 << SENSOR_BITS) - 1);

// ---------------------------------------------------------------------------
// Acoustic Sensor — hardware-accommodative driver
// ---------------------------------------------------------------------------
class AcousticSensor {
public:
  bool begin() {
    pinMode(SENSOR_PIN, INPUT);
    // Take a quick baseline ("dark" / ambient)
    baseline_ = rawAverage();
    Serial.print(F("[Sensor] baseline = "));
    Serial.println(baseline_, 1);
    return true;
  }

  /** Returns normalized observed energy in [0, 1]. */
  float readNormalized() {
    float raw = rawAverage();
    float delta = fabsf(raw - baseline_);

    // Simple soft scaling — adjust gain for your particular mic / receiver
    float gain = 4.5f;
    float norm = constrain(delta * gain / SENSOR_MAX_COUNTS, 0.0f, 1.0f);
    return norm;
  }

  /** Re-measure ambient baseline (call when emitters are off). */
  void recalibrateBaseline() {
    allQuietHint();
    delay(30);
    baseline_ = rawAverage();
    Serial.print(F("[Sensor] new baseline = "));
    Serial.println(baseline_, 1);
  }

  float baseline() const { return baseline_; }

private:
  float baseline_ = 0.0f;

  float rawAverage() {
    uint32_t sum = 0;
    for (int i = 0; i < SENSOR_SAMPLES; i++) {
      sum += analogRead(SENSOR_PIN);
      delayMicroseconds(40);
    }
    return (float)sum / (float)SENSOR_SAMPLES;
  }

  void allQuietHint() {
    // Placeholder: caller should ensure emitters are off before baseline
  }
};

// ---------------------------------------------------------------------------
// Ultrasonic driver (emitters)
// ---------------------------------------------------------------------------
class UltrasonicDriver {
public:
  bool begin() {
    for (int i = 0; i < NUM_EMITTERS; i++) {
      ledcSetup(PWM_CHANNELS[i], BASE_FREQ_HZ, PWM_RESOLUTION);
      ledcAttachPin(TRANSDUCER_PINS[i], PWM_CHANNELS[i]);
      ledcWrite(PWM_CHANNELS[i], 0);
    }
    return true;
  }

  void allOff() {
    for (int i = 0; i < NUM_EMITTERS; i++)
      ledcWrite(PWM_CHANNELS[i], 0);
  }

  void fire(uint16_t id, float amplitude = 0.7f, float freq_hz = BASE_FREQ_HZ) {
    if (id >= NUM_EMITTERS) return;
    allOff();
    ledcSetup(PWM_CHANNELS[id], (uint32_t)freq_hz, PWM_RESOLUTION);
    ledcAttachPin(TRANSDUCER_PINS[id], PWM_CHANNELS[id]);
    uint32_t duty = (uint32_t)(constrain(amplitude, 0.0f, 1.0f) * 255.0f);
    ledcWrite(PWM_CHANNELS[id], duty);
  }

  int numEmitters() const { return NUM_EMITTERS; }
};

// ---------------------------------------------------------------------------
// Echo Body
// ---------------------------------------------------------------------------
enum class RunMode : uint8_t { Passive, Held };

class EchoBody {
public:
  EchoBody(const char* node_id) {
    strncpy(node_id_, node_id, sizeof(node_id_) - 1);
    node_id_[sizeof(node_id_) - 1] = '\0';
  }

  bool begin() {
    Serial.println(F("[EchoBody] drivers + sensor"));
    if (!driver_.begin()) return false;
    if (!sensor_.begin()) return false;
    geometry_state_ = "uncalibrated";
    health_ = "ok";
    return true;
  }

  void exciteOnce(uint16_t id) {
    driver_.fire(id, 0.75f, BASE_FREQ_HZ);
    last_excitation_id_ = (int)id;

    // Let the acoustic field build, then sample
    delay(12);
    float observed = sensor_.readNormalized();
    emitObservation(observed);
  }

  void runSelfMap() {
    Serial.println(F("[EchoBody] MAP — acoustic baseline + response"));
    driver_.allOff();
    delay(20);
    sensor_.recalibrateBaseline();

    // One-hot response check (very light)
    for (int i = 0; i < driver_.numEmitters(); i++) {
      driver_.fire(i, 0.5f, BASE_FREQ_HZ);
      delay(15);
      float r = sensor_.readNormalized();
      Serial.print(F("  emitter ")); Serial.print(i);
      Serial.print(F(" → ")); Serial.println(r, 3);
      driver_.allOff();
      delay(10);
    }

    geometry_state_ = "calibrated";
    emitObservation(sensor_.readNormalized());
    Serial.println(F("[EchoBody] MAP done"));
  }

  void verifyIdentity(bool* unchanged) {
    *unchanged = (strcmp(geometry_state_, "calibrated") == 0);
    Serial.print(F("[EchoBody] VERIFY → "));
    Serial.println(*unchanged ? "trusted" : "needs MAP");
  }

  void tickPassive() {
    static uint32_t last = 0;
    if (millis() - last < 350) return;
    last = millis();

    float observed = sensor_.readNormalized();
    emitObservation(observed);
  }

  void allOff() {
    driver_.allOff();
    last_excitation_id_ = -1;
  }

  int numEmitters() const { return driver_.numEmitters(); }

private:
  void emitObservation(float observed) {
    Serial.print(F("OBS {"));
    Serial.print(F("\"body_id\":\"")); Serial.print(node_id_); Serial.print(F("\","));
    Serial.print(F("\"body_type\":\"ultrasonic\","));
    Serial.print(F("\"excitation_id\":")); Serial.print(last_excitation_id_); Serial.print(F(","));
    Serial.print(F("\"geometry_state\":\"")); Serial.print(geometry_state_); Serial.print(F("\","));
    Serial.print(F("\"health\":\"")); Serial.print(health_); Serial.print(F("\","));
    Serial.print(F("\"regions\":[{\"region\":\"acoustic\",\"observed\":"));
    Serial.print(observed, 4);
    Serial.print(F(",\"confidence\":0.75}]"));
    Serial.println(F("}"));
  }

  char node_id_[24];
  UltrasonicDriver driver_;
  AcousticSensor sensor_;
  const char* geometry_state_ = "uncalibrated";
  const char* health_ = "ok";
  int last_excitation_id_ = -1;
};

// ---------------------------------------------------------------------------
EchoBody body(ECHO_BODY_NODE_ID);
RunMode mode = RunMode::Passive;

bool pollCommand(String& cmd, int& arg) {
  if (!Serial.available()) return false;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return false;

  if (line.startsWith("EXCITE")) { cmd = "EXCITE"; arg = line.substring(6).toInt(); return true; }
  if (line == "MAP")     { cmd = "MAP"; return true; }
  if (line == "VERIFY")  { cmd = "VERIFY"; return true; }
  if (line == "PASSIVE") { cmd = "PASSIVE"; return true; }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(400);

  Serial.println(F("========================================"));
  Serial.println(F("  Echo Body  —  hardware-ready sensing"));
  Serial.println(F("  Sensor pin: ADC (default GPIO 34)"));
  Serial.println(F("========================================"));

  if (!body.begin()) {
    Serial.println(F("[FATAL] begin failed"));
    while (true) delay(1000);
  }

  bool unchanged = false;
  body.verifyIdentity(&unchanged);
  if (!unchanged) body.runSelfMap();

  Serial.println(F("[Boot] passive + live acoustic sensing ready"));
}

void loop() {
  String cmd;
  int arg = -1;

  if (pollCommand(cmd, arg)) {
    if (cmd == "EXCITE") {
      if (arg >= 0 && arg < body.numEmitters()) {
        body.exciteOnce((uint16_t)arg);
        mode = RunMode::Held;
      }
    } else if (cmd == "MAP") {
      body.runSelfMap();
    } else if (cmd == "VERIFY") {
      bool ok = false;
      body.verifyIdentity(&ok);
    } else if (cmd == "PASSIVE") {
      body.allOff();
      mode = RunMode::Passive;
    }
  }

  if (mode == RunMode::Passive) {
    body.tickPassive();
    delay(20);
  } else {
    delay(25);
  }
}
