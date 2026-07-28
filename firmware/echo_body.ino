/**
 * Echo Body — Ultrasonic physical body
 *
 * Implements the shared Field Body Protocol (see docs/FIELD_BODY_PROTOCOL.md).
 * Architectural sibling of optical-body-s3.
 *
 * Commands:
 *   EXCITE <id>
 *   MAP
 *   VERIFY
 *   PASSIVE
 */

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#ifndef ECHO_BODY_NODE_ID
#define ECHO_BODY_NODE_ID "echo_us_001"
#endif

const int NUM_EMITTERS = 4;
const int TRANSDUCER_PINS[NUM_EMITTERS] = {25, 26, 27, 14};
const int PWM_CHANNELS[NUM_EMITTERS]    = {0, 1, 2, 3};
const int PWM_RESOLUTION = 8;
const int BASE_FREQ_HZ   = 40000;

// ---------------------------------------------------------------------------
// Ultrasonic driver
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
// Echo Body (implements Field Body Protocol)
// ---------------------------------------------------------------------------
enum class RunMode : uint8_t { Passive, Held };

class EchoBody {
public:
  EchoBody(const char* node_id) {
    strncpy(node_id_, node_id, sizeof(node_id_) - 1);
    node_id_[sizeof(node_id_) - 1] = '\0';
  }

  bool begin() {
    Serial.println(F("[EchoBody] init ultrasonic drivers"));
    if (!driver_.begin()) return false;
    geometry_state_ = "uncalibrated";
    health_ = "ok";
    return true;
  }

  void exciteOnce(uint16_t id) {
    Serial.print(F("[EchoBody] EXCITE "));
    Serial.println(id);
    driver_.fire(id, 0.75f, BASE_FREQ_HZ);
    last_excitation_id_ = id;
  }

  void runSelfMap() {
    Serial.println(F("[EchoBody] MAP (acoustic calibration stub)"));
    // Future: one-hot sequence + receiver capture → transfer matrix
    geometry_state_ = "calibrated";
    Serial.println(F("[EchoBody] MAP done"));
  }

  void verifyIdentity(bool* unchanged) {
    *unchanged = (strcmp(geometry_state_, "calibrated") == 0);
    Serial.print(F("[EchoBody] VERIFY → "));
    Serial.println(*unchanged ? "trusted" : "needs MAP");
  }

  void tickPassive() {
    // Future: sparse acoustic observation packets
  }

  void allOff() {
    driver_.allOff();
    last_excitation_id_ = -1;
  }

  int numEmitters() const { return driver_.numEmitters(); }
  const char* nodeId() const { return node_id_; }
  const char* geometryState() const { return geometry_state_; }
  const char* health() const { return health_; }
  int lastExcitation() const { return last_excitation_id_; }

private:
  char node_id_[24];
  UltrasonicDriver driver_;
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

  if (line.startsWith("EXCITE")) {
    cmd = "EXCITE";
    arg = line.substring(6).toInt();
    return true;
  }
  if (line == "MAP")     { cmd = "MAP";     return true; }
  if (line == "VERIFY")  { cmd = "VERIFY";  return true; }
  if (line == "PASSIVE") { cmd = "PASSIVE"; return true; }

  Serial.print(F("[CMD] unknown: "));
  Serial.println(line);
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println(F("========================================"));
  Serial.println(F("  Echo Body  —  Ultrasonic Field Body"));
  Serial.println(F("  Protocol : Field Body v0.1"));
  Serial.println(F("  cmds     : EXCITE <id> | MAP | VERIFY | PASSIVE"));
  Serial.println(F("========================================"));
  Serial.print(F("Node ID   : "));
  Serial.println(ECHO_BODY_NODE_ID);

  if (!body.begin()) {
    while (true) delay(1000);
  }

  bool unchanged = false;
  body.verifyIdentity(&unchanged);
  if (!unchanged) {
    body.runSelfMap();
  }

  Serial.println(F("[Boot] passive ready"));
}

void loop() {
  String cmd;
  int arg = -1;

  if (pollCommand(cmd, arg)) {
    if (cmd == "EXCITE") {
      if (arg < 0 || arg >= body.numEmitters()) {
        Serial.println(F("[CMD] id out of range"));
      } else {
        body.exciteOnce((uint16_t)arg);
        mode = RunMode::Held;
      }
    }
    else if (cmd == "MAP") {
      body.runSelfMap();
    }
    else if (cmd == "VERIFY") {
      bool ok = false;
      body.verifyIdentity(&ok);
    }
    else if (cmd == "PASSIVE") {
      body.allOff();
      mode = RunMode::Passive;
      Serial.println(F("[CMD] PASSIVE"));
    }
  }

  if (mode == RunMode::Passive) {
    body.tickPassive();
    delay(200);
  } else {
    delay(40);
  }
}
