/*
 * Echo Grid Ultrasonic OS — ESP32 Swarm Node
 * Receives phase/amp/freq packets and drives a local ultrasonic transducer.
 * Uses ESP-NOW for low-latency distributed synchronization.
 */

#include <esp_now.h>
#include <WiFi.h>

const int TRANSDUCER_PIN = 25;   // PWM-capable pin
const int PWM_CHANNEL    = 0;
const int PWM_RES        = 8;    // 8-bit duty
const int BASE_FREQ      = 40000;

// Simple packet structure (expand as needed)
typedef struct {
  uint8_t x;
  uint8_t y;
  float   freq;
  float   amp;
  float   phase;
} EchoPacket;

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
  if (len < sizeof(EchoPacket)) return;

  EchoPacket pkt;
  memcpy(&pkt, incomingData, sizeof(pkt));

  // Update local PWM frequency and duty
  ledcSetup(PWM_CHANNEL, (uint32_t)pkt.freq, PWM_RES);
  ledcAttachPin(TRANSDUCER_PIN, PWM_CHANNEL);

  uint32_t duty = (uint32_t)(pkt.amp * 255.0f);
  ledcWrite(PWM_CHANNEL, duty);

  // Optional: Serial debug
  // Serial.printf("Node update: f=%.1f  A=%.2f\n", pkt.freq, pkt.amp);
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);

  // Default 40 kHz idle
  ledcSetup(PWM_CHANNEL, BASE_FREQ, PWM_RES);
  ledcAttachPin(TRANSDUCER_PIN, PWM_CHANNEL);
  ledcWrite(PWM_CHANNEL, 0);

  Serial.println("Echo Grid ESP32 node ready");
}

void loop() {
  // Heartbeat / local diagnostics can go here
  delay(100);
}
