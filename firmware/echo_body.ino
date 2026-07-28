/**
 * Echo Body — Ultrasonic physical body for Echo Grid / MetaField
 *
 * Architectural sibling of optical-body-s3.
 * Same command language, same observation philosophy,
 * different physics (sound instead of light).
 *
 * Commands (Serial 115200):
 *   EXCITE <id>   — shape a specific ultrasonic emitter
 *   MAP           — run acoustic self-map / calibration
 *   VERIFY        — identity + health check
 *   PASSIVE       — resume background observation
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#ifndef ECHO_BODY_NODE_ID
#define ECHO_BODY_NODE_ID "echo_us_001"
#endif

const int NUM_EMITTERS     = 4;          // start small; scale later
const int TRANSDUCER_PINS[NUM_EMITTERS] = {25, 26, 27, 14};  // change to your wiring
const int PWM_CHANNELS[NUM_EMITTERS]    = {0, 1, 2, 3};
const int PWM_RESOLUTION  = 8;
const int BASE_FREQ_HZ    = 40000;

// ---------------------------------------------------------------------------
// Simple FieldObservation (ultrasonic flavour)
// ---------------------------------------------------------------------------
struct FieldRegion {
  char   region[16];
  float  observed;     // 0..1
  float  confidence;
};

struct FieldObservation {
  char   body_id[24];
  char   body_type[16];      // "ultrasonic"
  int32_t excitation_id;
  FieldRegion regions[8];
  int    num_regions;
  char   geometry_state[16]; // uncalibrated | calibrated | degraded
  char   health[8];          // ok | partial | error
  int    schema_version = 1;
};

// ---------------------------------------------------------------------------
// Ultrasonic driver (LEDC)
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
    for (int i = 0; i < NUM_EMITTERS; i++) {
      ledcWrite(PWM_CHANNELS[i], 0);
    }
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
  }

  bool begin() {
    Serial.println(F("[EchoBody] initialising ultrasonic drivers..."));
    if (!driver_.begin()) {
      Serial.println(F("[FATAL] UltrasonicDriver failed"));
      return false;
    }
    geometry_state_ = "uncalibrated";
    health_ = "ok";
    return true;
  }

  void exciteOnce(uint16_t id) {
    Serial.print(F("[EchoBody] EXCITE emitter "));
    Serial.println(id);
    driver_.fire(id, 0.75f, BASE_FREQ_HZ);
    // Hold for a short acoustic burst; caller decides longer duration
    delay(30);
    // leave it running in Held mode; PASSIVE will silence
  }

  void runSelfMap() {
    Serial.println(F("[EchoBody] MAP — acoustic self-calibration (placeholder)"));
    // TODO: one-hot excitation + microphone / receiver capture
    //       build simple transfer / fingerprint
    geometry_state_ = "calibrated";
    Serial.println(F("[EchoBody] MAP complete (stub)"));
  }

  void verifyIdentity(bool* unchanged) {
    // Placeholder: later compare acoustic fingerprint in FRAM / NVS
    *unchanged = (strcmp(geometry_state_, "calibrated") == 0);
    Serial.print(F("[EchoBody] VERIFY → "));
    Serial.println(*unchanged ? "trusted" : "needs map");
  }

  void tickPassive() {
    // Background field observation would go here
    // (listen with a receiver, emit sparse FieldObservation packets)
  }

  void allOff() { driver_.allOff(); }

  int numEmitters() const { return driver_.numEmitters(); }

  const char* geometryState() const { return geometry_state_; }
  const char* health() const { return health_; }

private:
  char node_id_[24];
  UltrasonicDriver driver_;
  const char* geometry_state_ = "uncalibrated";
  const char* health_ = "ok";
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
EchoBody body(ECHO_BODY_NODE_ID);
RunMode mode = RunMode::Passive;

// Very simple command parser (Serial)
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

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(600);

  Serial.println();
  Serial.println(F("========================================"));
  Serial.println(F("  Echo Body  —  Ultrasonic physical body"));
  Serial.println(F("  cmds: EXCITE <id> | MAP | VERIFY | PASSIVE"));
  Serial.println(F("========================================"));
  Serial.print(F("Node ID : "));
  Serial.println(ECHO_BODY_NODE_ID);

  if (!body.begin()) {
    while (true) delay(1000);
  }

  bool unchanged = false;
  body.verifyIdentity(&unchanged);
  if (!unchanged) {
    Serial.println(F("[Boot] running MAP..."));
    body.runSelfMap();
  } else {
    Serial.println(F("[Boot] identity trusted"));
  }

  Serial.println(F("[Boot] passive loop ready"));
}

void loop() {
  String cmd;
  int arg = -1;

  if (pollCommand(cmd, arg)) {
    if (cmd == "EXCITE") {
      if (arg < 0 || arg >= body.numEmitters()) {
        Serial.println(F("[CMD] EXCITE id out of range"));
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
      Serial.println(F("[CMD] passive resume"));
      body.allOff();
      mode = RunMode::Passive;
    }
  }

  if (mode == RunMode::Passive) {
    body.tickPassive();
    delay(200);
  } else {
    delay(40);  // Held: keep emitter alive until PASSIVE
  }
}
