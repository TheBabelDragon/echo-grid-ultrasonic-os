/**
 * Echo Body — Ultrasonic physical body (closed-loop capable)
 *
 * Implements Field Body Protocol + emits observation lines.
 *
 * Observation format (body → host):
 *   OBS {"body_id":"echo_us_001","body_type":"ultrasonic","excitation_id":-1,"geometry_state":"calibrated","health":"ok","regions":[{"region":"ambient","observed":0.12,"confidence":0.5}]}
 */

#include <Arduino.h>

#ifndef ECHO_BODY_NODE_ID
#define ECHO_BODY_NODE_ID "echo_us_001"
#endif

const int NUM_EMITTERS = 4;
const int TRANSDUCER_PINS[NUM_EMITTERS] = {25, 26, 27, 14};
const int PWM_CHANNELS[NUM_EMITTERS]    = {0, 1, 2, 3};
const int PWM_RESOLUTION = 8;
const int BASE_FREQ_HZ   = 40000;

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
    for (int i = 0; i < NUM_EMITTERS; i++) ledcWrite(PWM_CHANNELS[i], 0);
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

enum class RunMode : uint8_t { Passive, Held };

class EchoBody {
public:
  EchoBody(const char* node_id) {
    strncpy(node_id_, node_id, sizeof(node_id_) - 1);
    node_id_[sizeof(node_id_) - 1] = '\0';
  }

  bool begin() {
    if (!driver_.begin()) return false;
    geometry_state_ = "uncalibrated";
    health_ = "ok";
    return true;
  }

  void exciteOnce(uint16_t id) {
    driver_.fire(id, 0.75f, BASE_FREQ_HZ);
    last_excitation_id_ = (int)id;
    // Simple synthetic observation after excitation
    emitObservation(0.55f + 0.2f * (id % 3));
  }

  void runSelfMap() {
    Serial.println(F("[EchoBody] MAP"));
    geometry_state_ = "calibrated";
    emitObservation(0.1f);
  }

  void verifyIdentity(bool* unchanged) {
    *unchanged = (strcmp(geometry_state_, "calibrated") == 0);
    Serial.print(F("[EchoBody] VERIFY → "));
    Serial.println(*unchanged ? "trusted" : "needs MAP");
  }

  void tickPassive() {
    // Emit a low-level ambient observation so the host has a closed-loop signal
    static uint32_t last = 0;
    if (millis() - last > 800) {
      last = millis();
      float ambient = 0.08f + 0.04f * sin(millis() / 1500.0f);
      emitObservation(ambient);
    }
  }

  void allOff() {
    driver_.allOff();
    last_excitation_id_ = -1;
  }

  int numEmitters() const { return driver_.numEmitters(); }

private:
  void emitObservation(float observed) {
    // Compact single-line observation the host can parse
    Serial.print(F("OBS {"));
    Serial.print(F("\"body_id\":\"")); Serial.print(node_id_); Serial.print(F("\","));
    Serial.print(F("\"body_type\":\"ultrasonic\","));
    Serial.print(F("\"excitation_id\":")); Serial.print(last_excitation_id_); Serial.print(F(","));
    Serial.print(F("\"geometry_state\":\"")); Serial.print(geometry_state_); Serial.print(F("\","));
    Serial.print(F("\"health\":\"")); Serial.print(health_); Serial.print(F("\","));
    Serial.print(F("\"regions\":[{\"region\":\"ambient\",\"observed\":"));
    Serial.print(observed, 3);
    Serial.print(F(",\"confidence\":0.6}]"));
    Serial.println(F("}"));
  }

  char node_id_[24];
  UltrasonicDriver driver_;
  const char* geometry_state_ = "uncalibrated";
  const char* health_ = "ok";
  int last_excitation_id_ = -1;
};

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
  Serial.println(F("  Echo Body  —  closed-loop ultrasonic"));
  Serial.println(F("  Protocol : Field Body v0.1 + OBS"));
  Serial.println(F("========================================"));

  if (!body.begin()) while (true) delay(1000);

  bool unchanged = false;
  body.verifyIdentity(&unchanged);
  if (!unchanged) body.runSelfMap();

  Serial.println(F("[Boot] passive + observation stream ready"));
}

void loop() {
  String cmd; int arg = -1;
  if (pollCommand(cmd, arg)) {
    if (cmd == "EXCITE") {
      if (arg >= 0 && arg < body.numEmitters()) {
        body.exciteOnce((uint16_t)arg);
        mode = RunMode::Held;
      }
    } else if (cmd == "MAP") {
      body.runSelfMap();
    } else if (cmd == "VERIFY") {
      bool ok = false; body.verifyIdentity(&ok);
    } else if (cmd == "PASSIVE") {
      body.allOff();
      mode = RunMode::Passive;
    }
  }

  if (mode == RunMode::Passive) {
    body.tickPassive();
    delay(50);
  } else {
    delay(30);
  }
}
