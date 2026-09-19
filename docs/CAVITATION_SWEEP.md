# CAVITATION.SWEEP

Measurement-first cavitation calibration subsystem for Echo Grid Ultrasonic OS.

**Location:** `echo_grid/cavitation.py`

This subsystem characterises the relationship between an abstract ultrasonic
excitation *command* and the measured acoustic / optical response. It does
**not** assume that ESP/PWM drive level equals acoustic pressure.

> **Safety notice**  
> The software does **not** determine safe physical acoustic limits. Those
> limits must come from the transducer / chamber / sensor calibration and
> applicable experimental procedures. All operating limits are explicit
> configuration values supplied by the experiment operator.

---

## Experiment state machine

```
IDLE
  → BASELINE
  → DRIVE_STEP
  → SETTLE
  → MEASURE
  → CLASSIFY
  → NEXT_STEP
  → COMPLETE

Also reachable at any time:
  ABORT   (operator request)
  FAULT   (interlock trip — fail closed)
```

| State | Action |
|-------|--------|
| IDLE | Sweep constructed, not started |
| BASELINE | Collect optical, acoustic (passive), temperature; compute noise statistics |
| DRIVE_STEP | Apply abstract `drive_command` for configured burst duration |
| SETTLE | Wait for transient decay |
| MEASURE | Aggregate multi-sample sensor window |
| CLASSIFY | Run conservative onset detector |
| NEXT_STEP | Advance drive grid or enter bounded characterisation |
| COMPLETE | Normal termination |
| ABORT / FAULT | Drive forced to zero; experiment stopped |

---

## Sensor inputs

Abstracted behind `SensorSource`:

| Channel | Role |
|---------|------|
| `acoustic_rms` | Broadband / band-limited acoustic energy |
| `acoustic_transient` | High-frequency / impulsive content |
| `measured_pressure` | Optional calibrated pressure (external layer) |
| `optical_total` / `optical_peak` / `optical_centroid` / `optical_latency` | Photon / scatter response |
| `temperature` | Thermal interlock |

Hardware implementations plug in by subclassing `SensorSource`. Waveform and
device-specific details remain behind the existing ultrasonic / body interfaces.

---

## Baseline methodology

Before any excitation:

1. Force drive to zero.
2. Collect `baseline_samples` (default 32) of acoustic and optical channels.
3. Record temperature and timestamp.
4. Compute mean and standard deviation; enforce a minimum std floor so that
   zero-variance baselines do not produce infinite σ scores.
5. **Never** declare cavitation from a single anomalous sample.

---

## Onset classification

Implemented by `CavitationDetector`.

Rules (conservative by design):

- Optical intensity alone is **never** sufficient.
- Prefer the causal chain:  
  **ultrasonic excitation → acoustic response → optical response**.
- A candidate requires evidence above baseline by configured σ thresholds.
- Confirmation requires `min_repeat_count` independent observations that also
  satisfy a confidence floor.

Status progression:

```
NONE → CAVITATION_ONSET_CANDIDATE → CAVITATION_ONSET_CONFIRMED
```

### Bounded characterisation

After a confirmed onset the sweep does **not** keep increasing drive
indefinitely. It records a small number of points around the transition
(`characterize_steps`, `max_drive_above_onset`) and then terminates.

---

## Safety / interlock model

Mandatory software-side checks (any failure → drive = 0 → FAULT or ABORT):

| Interlock | Config key |
|-----------|------------|
| Maximum drive command | `max_drive_command` |
| Maximum burst duration | `max_burst_duration_s` |
| Maximum experiment duration | `max_experiment_duration_s` |
| Maximum temperature | `max_temperature_c` |
| Sensor-loss / invalid reading | `sensor_timeout_s` / `valid=False` |
| Invalid calibration / config | raised at construction |
| Emergency / manual abort | `request_abort()` |
| Device-reported fault | propagated from `SensorReading.fault` |

The subsystem **fails closed**.

Default configuration values are intentionally conservative. Raising limits
is an explicit operator action.

---

## JSON event format

```json
{
  "event_type": "CAVITATION_EVENT",
  "excitation_id": "<sweep_id>:<point_index>",
  "frequency_hz": 40000.0,
  "drive_command": 0.24,
  "measured_pressure": 1.85,
  "onset_confidence": 0.78,
  "acoustic_signature": {
    "rms": 0.154,
    "transient": 0.21,
    "sigma": 9.2
  },
  "photon_signature": {
    "total": 0.12,
    "peak": 0.18,
    "centroid": 0.41,
    "latency": 0.009,
    "sigma": 6.1
  },
  "timestamp": "2026-09-19T01:00:00+00:00",
  "sweep_id": "...",
  "point_index": 8,
  "onset_status": "CAVITATION_ONSET_CONFIRMED"
}
```

No biological interpretation is present in this schema.

Each measurement point (`CavitationSweepPoint`) additionally records:

`sweep_id`, `point_index`, `frequency_hz`, `drive_command`, `burst_duration`,
`timestamp`, `measured_pressure`, `acoustic_rms`, `acoustic_transient`,
`optical_total`, `optical_peak`, `optical_centroid`, `optical_latency`,
`temperature`, `baseline_deviation`, `onset_status`, `onset_confidence`.

---

## MetaField integration

Sweep measurements are exposed as additional regions on the existing
`FieldObservation` path (via `CavitationSweep.to_metafield_regions`):

- `acoustic`
- `photon_total`
- `photon_peak`
- `photon_centroid`
- `cavitation_onset`
- `cavitation_confidence`

Compatible with the MetaField JSONL bridge; does not introduce a competing
telemetry protocol.

---

## Simulation mode

**Mandatory and the default.**

`SimulatedSensorSource` provides a deterministic sensor model so that
`CAVITATION.SWEEP` can execute without a transducer, tank, or high-power
hardware.

```bash
python tools/cavitation_sweep_demo.py
# or
python -m echo_grid.cavitation   # if __main__ entry added
```

Hardware drive path (if ever wired) requires an explicit opt-in flag
`--hardware`. Simulation remains the default.

---

## Hardware integration boundary

| Layer | Responsibility |
|-------|----------------|
| `CavitationSweep` / detector | State machine, interlocks, onset logic, records |
| `SensorSource` | Drive command & multi-modal readout |
| Existing body / ultrasonic interfaces | Waveform generation, ESP/PWM, FPGA DDS |
| External calibration | Maps `acoustic_rms` / sensor volts → physical pressure |

Connecting calibrated acoustic + optical sensors later does **not** require
redesigning the sweep architecture — only a new `SensorSource` implementation.

---

## CLI / demo

```bash
# pure simulation (default)
python tools/cavitation_sweep_demo.py

# print response curve only
python tools/cavitation_sweep_demo.py --quiet

# custom limits (still simulated)
python tools/cavitation_sweep_demo.py --drive-stop 0.35 --step 0.02
```

Never energises physical hardware unless `--hardware` is supplied **and** a
real sensor backend is registered.

---

## Acceptance

A user can run a completely hardware-free `CAVITATION.SWEEP` simulation,
obtain a reproducible response curve and onset event, and later connect
calibrated acoustic + optical sensors without redesigning the sweep
architecture.
