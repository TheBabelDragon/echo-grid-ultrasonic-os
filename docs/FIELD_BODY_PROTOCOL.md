# Field Body Protocol (v0.1)

Common contract between any physical body (optical, ultrasonic, future modalities) and a host (MetaField, Echo Grid OS, etc.).

The body is **not** the intelligence.  
It only:
- accepts excitation commands
- reports observations
- maintains a simple identity / calibration state

---

## 1. Commands (host → body)

All commands are line-oriented, ASCII, terminated by `\n`.

| Command | Meaning | Example |
|---------|---------|---------|
| `EXCITE <id>` | Shape / fire one emitter or source | `EXCITE 2` |
| `MAP` | Run self-calibration / transfer map | `MAP` |
| `VERIFY` | Check identity + health | `VERIFY` |
| `PASSIVE` | Return to background observation mode | `PASSIVE` |

Optional future commands (reserved):
- `SET <param> <value>`
- `QUERY STATUS`

---

## 2. FieldObservation (body → host)

Minimal shared shape (JSON or binary later).

```json
{
  "body_id": "echo_us_001",
  "body_type": "ultrasonic",          // or "optical"
  "excitation_id": 2,                 // -1 if passive
  "geometry_state": "calibrated",     // uncalibrated | calibrating | calibrated | degraded
  "health": "ok",                     // ok | partial | error
  "schema_version": 1,
  "regions": [
    {
      "region": "emitter_02",
      "observed": 0.73,
      "confidence": 0.91
    }
  ],
  "modality": {
    // modality-specific payload
    // ultrasonic example:
    "freq_hz": 40120,
    "amplitude": 0.7,
    "phase": 0.12
  }
}
```

### Rules
- `body_type` distinguishes the physics.
- `regions[]` is the common observation surface.
- `modality` is free-form and owned by the specific body.
- Hosts should ignore unknown modality fields.

---

## 3. Runtime modes

```
Passive  → body observes / reports sparsely, no strong excitation
Held     → body is under active EXCITE control until PASSIVE
```

---

## 4. Identity

Every body should be able to answer:

- Who am I? (`body_id`)
- Am I still the same physical object? (`VERIFY` + stored fingerprint)
- Do I need recalibration? (`geometry_state`)

Storage can be FRAM, NVS, or SD — the protocol does not dictate the medium.

---

## 5. Design intent

- Optical body and Ultrasonic (Echo) body are **siblings**.
- MetaField (or any host) talks to either through the same command + observation contract.
- New modalities (magnetic, haptic, RF…) only need to implement this protocol.

This document is the stable architectural hinge between the different physical bodies.
