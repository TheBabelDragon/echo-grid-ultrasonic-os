# Echo Body — Ultrasonic Physical Body for Echo Grid / MetaField

This firmware is the **ultrasonic** counterpart to `optical-body-s3`.

```
Echo Body (ESP32)
        │
        │  excitation + FieldObservation packets
        ↓
MetaField / Echo Grid host
```

It follows the same architectural principles as the optical body:

- Node identity
- Command language: `EXCITE <id> | MAP | VERIFY | PASSIVE`
- Passive vs Held modes
- FieldObservation packets (modality-adapted)
- Clean driver / protocol / body separation

## Hardware target (Phase 0)

- ESP32 (or ESP32-S3)
- One or more 40 kHz ultrasonic transducers driven via MOSFET + LEDC PWM
- Optional: microphone or ultrasonic receiver for closed-loop later

## Commands (Serial 115200)

```
EXCITE <id>     Fire / shape a specific emitter
MAP             Run self-calibration (acoustic fingerprint)
VERIFY          Check identity / health
PASSIVE         Return to background field observation
```

## Build

Use Arduino IDE or PlatformIO.  
Pin map and transducer count are defined at the top of `echo_body.ino`.
