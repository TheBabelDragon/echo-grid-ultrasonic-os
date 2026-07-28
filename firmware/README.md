# Echo Body

Ultrasonic physical body for Echo Grid / MetaField.

Implements the shared **Field Body Protocol**  
→ see [`docs/FIELD_BODY_PROTOCOL.md`](../docs/FIELD_BODY_PROTOCOL.md)

## Relationship to optical-body-s3

| Aspect              | optical-body-s3      | Echo Body              |
|---------------------|----------------------|------------------------|
| Modality            | Light (laser + PD)   | Sound (ultrasonic)     |
| Protocol            | Same spirit          | Explicit v0.1 contract |
| Commands            | EXCITE/MAP/VERIFY/PASSIVE | Identical         |
| Role                | Physical body only   | Physical body only     |

They are siblings. MetaField (or any host) can talk to either through the same command + observation surface.

## Quick start

1. Set transducer pins in `echo_body.ino`
2. Flash to ESP32
3. Serial 115200
4. Try:
   ```
   EXCITE 0
   EXCITE 1
   PASSIVE
   MAP
   VERIFY
   ```
