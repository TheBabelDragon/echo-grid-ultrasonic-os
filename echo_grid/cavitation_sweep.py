"""CAVITATION.SWEEP detector and sweep runner."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cavitation_sensors import (
    CavitationState,
    OnsetStatus,
    CavitationSweepConfig,
    CavitationSweepPoint,
    CavitationEvent,
    SensorReading,
    SensorSource,
    SimulatedSensorSource,
)


class CavitationDetector:
    """
    Conservative multi-evidence onset detector.

    - Never declares cavitation from a single anomalous sample.
    - Never declares cavitation from optical intensity alone.
    - Prefers ultrasonic excitation → acoustic response → optical response.
    - Emits CANDIDATE then CONFIRMED only after configured repeatability.
    """

    def __init__(self, config: CavitationSweepConfig):
        self.cfg = config
        self.baseline_acoustic_mean = 0.0
        self.baseline_acoustic_std = config.baseline_min_std_floor
        self.baseline_optical_mean = 0.0
        self.baseline_optical_std = config.baseline_min_std_floor
        self.candidate_count = 0
        self.confirmed = False
        self.last_confidence = 0.0
        self.onset_drive: Optional[float] = None

    def set_baseline(
        self,
        acoustic_samples: Sequence[float],
        optical_samples: Sequence[float],
    ) -> None:
        a = np.asarray(acoustic_samples, dtype=np.float64)
        o = np.asarray(optical_samples, dtype=np.float64)
        if len(a) == 0:
            a = np.array([0.0])
        if len(o) == 0:
            o = np.array([0.0])
        self.baseline_acoustic_mean = float(np.mean(a))
        self.baseline_acoustic_std = max(float(np.std(a)), self.cfg.baseline_min_std_floor)
        self.baseline_optical_mean = float(np.mean(o))
        self.baseline_optical_std = max(float(np.std(o)), self.cfg.baseline_min_std_floor)

    def evaluate(
        self,
        acoustic_rms: Optional[float],
        acoustic_transient: Optional[float],
        optical_total: Optional[float],
        drive_command: float,
    ) -> Tuple[OnsetStatus, float, Dict[str, Any]]:
        """Returns (status, confidence, evidence dict)."""
        evidence: Dict[str, Any] = {
            "acoustic_sigma": 0.0,
            "optical_sigma": 0.0,
            "acoustic_ok": False,
            "optical_ok": False,
        }

        if acoustic_rms is None:
            acoustic_rms = 0.0
        if acoustic_transient is None:
            acoustic_transient = 0.0
        if optical_total is None:
            optical_total = 0.0

        a_dev = acoustic_rms - self.baseline_acoustic_mean
        a_sigma = a_dev / self.baseline_acoustic_std
        t_sigma = (acoustic_transient - self.baseline_acoustic_mean) / self.baseline_acoustic_std
        o_dev = optical_total - self.baseline_optical_mean
        o_sigma = o_dev / self.baseline_optical_std

        evidence["acoustic_sigma"] = float(max(a_sigma, t_sigma))
        evidence["optical_sigma"] = float(o_sigma)

        acoustic_ok = evidence["acoustic_sigma"] >= self.cfg.acoustic_threshold_sigma
        optical_ok = evidence["optical_sigma"] >= self.cfg.optical_threshold_sigma
        evidence["acoustic_ok"] = acoustic_ok
        evidence["optical_ok"] = optical_ok

        if self.cfg.require_acoustic_for_onset and not acoustic_ok:
            self.last_confidence = 0.0
            return OnsetStatus.NONE, 0.0, evidence

        if not acoustic_ok and not optical_ok:
            self.last_confidence = 0.0
            return OnsetStatus.NONE, 0.0, evidence

        conf = 0.0
        if acoustic_ok:
            conf += 0.55 * min(1.0, evidence["acoustic_sigma"] / (self.cfg.acoustic_threshold_sigma * 1.5))
        if optical_ok and acoustic_ok:
            conf += 0.45 * min(1.0, evidence["optical_sigma"] / (self.cfg.optical_threshold_sigma * 1.5))
        conf = float(np.clip(conf, 0.0, 1.0))
        self.last_confidence = conf

        if conf < 0.35:
            return OnsetStatus.NONE, conf, evidence

        self.candidate_count += 1
        if self.onset_drive is None:
            self.onset_drive = drive_command

        if self.candidate_count >= self.cfg.min_repeat_count and conf >= 0.55:
            self.confirmed = True
            return OnsetStatus.CONFIRMED, conf, evidence

        return OnsetStatus.CANDIDATE, conf, evidence


class CavitationSweep:
    """
    State-machine driven CAVITATION.SWEEP experiment.

    States: IDLE → BASELINE → DRIVE_STEP → SETTLE → MEASURE → CLASSIFY
            → NEXT_STEP → COMPLETE
            (+ ABORT / FAULT)

    Fail-closed: any interlock forces drive to zero and transitions to FAULT/ABORT.
    """

    def __init__(
        self,
        config: Optional[CavitationSweepConfig] = None,
        sensor: Optional[SensorSource] = None,
        hardware: bool = False,
    ):
        self.cfg = config or CavitationSweepConfig()
        self.cfg.validate()
        self.hardware = bool(hardware)
        if sensor is not None:
            self.sensor = sensor
        else:
            self.sensor = SimulatedSensorSource(seed=self.cfg.rng_seed or 42)
        self.detector = CavitationDetector(self.cfg)

        self.sweep_id = str(uuid.uuid4())
        self.state = CavitationState.IDLE
        self.points: List[CavitationSweepPoint] = []
        self.events: List[CavitationEvent] = []
        self.drive_grid: List[float] = []
        self._idx = 0
        self._t_start = 0.0
        self._fault_reason: Optional[str] = None
        self._abort_requested = False
        self._baseline_acoustic: List[float] = []
        self._baseline_optical: List[float] = []
        self._onset_found = False
        self._characterize_remaining = 0

    def request_abort(self) -> None:
        self._abort_requested = True

    def run(self) -> Dict[str, Any]:
        """Execute the full sweep (blocking). Returns serializable summary."""
        self.state = CavitationState.IDLE
        self._t_start = time.monotonic()
        self._build_drive_grid()
        try:
            self._run_baseline()
            if self.state in (CavitationState.FAULT, CavitationState.ABORT):
                return self._summary()

            while self._idx < len(self.drive_grid):
                if self._abort_requested:
                    self._safe_abort("manual_abort")
                    break
                if self._check_experiment_timeout():
                    break
                self._run_one_step()
                if self.state in (CavitationState.FAULT, CavitationState.ABORT, CavitationState.COMPLETE):
                    break
                if self._onset_found and self.cfg.characterize_after_onset:
                    if self._characterize_remaining <= 0:
                        self.state = CavitationState.COMPLETE
                        break

            if self.state not in (CavitationState.FAULT, CavitationState.ABORT, CavitationState.COMPLETE):
                self.state = CavitationState.COMPLETE
        finally:
            try:
                self.sensor.zero_drive()
            except Exception:
                pass
        return self._summary()

    def to_metafield_regions(self, last_point: Optional[CavitationSweepPoint] = None) -> List[Dict[str, Any]]:
        """Extra regions for MetaField FieldObservation path (non-competing)."""
        pt = last_point or (self.points[-1] if self.points else None)
        if pt is None:
            return []
        regions = []

        def _reg(name: str, val: Optional[float], conf: float = 0.8, anomaly: float = 0.0, extras=None):
            if val is None:
                return
            regions.append({
                "region": name,
                "observed": float(np.clip(val, 0.0, 1.0)) if name.startswith("photon") or name == "acoustic" else float(val),
                "expected": None,
                "confidence": conf,
                "anomaly": anomaly,
                "extras": extras or {},
            })

        if pt.acoustic_rms is not None:
            _reg("acoustic", min(1.0, pt.acoustic_rms / 2.0), 0.85, extras={"rms": pt.acoustic_rms})
        if pt.optical_total is not None:
            _reg("photon_total", min(1.0, pt.optical_total / 2.0), 0.8)
        if pt.optical_peak is not None:
            _reg("photon_peak", min(1.0, pt.optical_peak / 2.5), 0.75)
        if pt.optical_centroid is not None:
            _reg("photon_centroid", pt.optical_centroid, 0.7)
        onset_flag = 1.0 if pt.onset_status != OnsetStatus.NONE.value else 0.0
        _reg(
            "cavitation_onset",
            onset_flag,
            0.9 if onset_flag else 0.5,
            anomaly=0.0 if pt.onset_status != OnsetStatus.CONFIRMED.value else 0.2,
            extras={"status": pt.onset_status},
        )
        _reg("cavitation_confidence", pt.onset_confidence, 0.85)
        return regions

    def _build_drive_grid(self) -> None:
        g = []
        d = self.cfg.drive_start
        while d <= self.cfg.drive_stop + 1e-12:
            if d > self.cfg.max_drive_command:
                break
            g.append(round(d, 6))
            d += self.cfg.drive_step
        if not g:
            g = [self.cfg.drive_start]
        self.drive_grid = g

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _check_interlocks(self, drive: float, burst: float, reading: Optional[SensorReading] = None) -> Optional[str]:
        if drive > self.cfg.max_drive_command + 1e-9:
            return "max_drive_command"
        if burst > self.cfg.max_burst_duration_s + 1e-9:
            return "max_burst_duration"
        if time.monotonic() - self._t_start > self.cfg.max_experiment_duration_s:
            return "max_experiment_duration"
        if reading is not None:
            if not reading.valid:
                return reading.fault or "sensor_loss"
            if reading.temperature is not None and reading.temperature > self.cfg.max_temperature_c:
                return "max_temperature"
        if self._abort_requested:
            return "manual_abort"
        return None

    def _safe_abort(self, reason: str) -> None:
        try:
            self.sensor.zero_drive()
        except Exception:
            pass
        self._fault_reason = reason
        if reason == "manual_abort":
            self.state = CavitationState.ABORT
        else:
            self.state = CavitationState.FAULT

    def _check_experiment_timeout(self) -> bool:
        if time.monotonic() - self._t_start > self.cfg.max_experiment_duration_s:
            self._safe_abort("max_experiment_duration")
            return True
        return False

    def _run_baseline(self) -> None:
        self.state = CavitationState.BASELINE
        self.sensor.zero_drive()
        acoustic = []
        optical = []
        for _ in range(self.cfg.baseline_samples):
            r = self.sensor.read()
            reason = self._check_interlocks(0.0, 0.0, r)
            if reason:
                self._safe_abort(reason)
                return
            if r.acoustic_rms is not None:
                acoustic.append(r.acoustic_rms)
            if r.optical_total is not None:
                optical.append(r.optical_total)
            time.sleep(0.002)
        self._baseline_acoustic = acoustic
        self._baseline_optical = optical
        self.detector.set_baseline(acoustic, optical)

    def _run_one_step(self) -> None:
        drive = self.drive_grid[self._idx]
        burst = self.cfg.burst_duration_s

        self.state = CavitationState.DRIVE_STEP
        reason = self._check_interlocks(drive, burst)
        if reason:
            self._safe_abort(reason)
            return

        try:
            self.sensor.set_drive(drive, self.cfg.frequency_hz, burst)
        except Exception as e:
            self._safe_abort(f"drive_error:{e}")
            return

        self.state = CavitationState.SETTLE
        time.sleep(min(self.cfg.settle_time_s, 0.05))

        self.state = CavitationState.MEASURE
        readings: List[SensorReading] = []
        t_end = time.monotonic() + min(self.cfg.measure_window_s, 0.08)
        while time.monotonic() < t_end:
            r = self.sensor.read()
            reason = self._check_interlocks(drive, burst, r)
            if reason:
                self.sensor.zero_drive()
                self._safe_abort(reason)
                return
            readings.append(r)
            time.sleep(0.002)

        self.sensor.zero_drive()

        def _mean(attr: str) -> Optional[float]:
            vals = [getattr(x, attr) for x in readings if getattr(x, attr) is not None]
            return float(np.mean(vals)) if vals else None

        acoustic_rms = _mean("acoustic_rms")
        acoustic_transient = _mean("acoustic_transient")
        measured_pressure = _mean("measured_pressure")
        optical_total = _mean("optical_total")
        optical_peak = _mean("optical_peak")
        optical_centroid = _mean("optical_centroid")
        optical_latency = _mean("optical_latency")
        temperature = _mean("temperature")

        base_dev = None
        if acoustic_rms is not None:
            base_dev = abs(acoustic_rms - self.detector.baseline_acoustic_mean)

        self.state = CavitationState.CLASSIFY
        status, conf, evidence = self.detector.evaluate(
            acoustic_rms, acoustic_transient, optical_total, drive
        )

        pt = CavitationSweepPoint(
            sweep_id=self.sweep_id,
            point_index=self._idx,
            frequency_hz=self.cfg.frequency_hz,
            drive_command=drive,
            burst_duration=burst,
            timestamp=self._now_iso(),
            measured_pressure=measured_pressure,
            acoustic_rms=acoustic_rms,
            acoustic_transient=acoustic_transient,
            optical_total=optical_total,
            optical_peak=optical_peak,
            optical_centroid=optical_centroid,
            optical_latency=optical_latency,
            temperature=temperature,
            baseline_deviation=base_dev,
            onset_status=status.value,
            onset_confidence=conf,
            notes=json.dumps(evidence),
        )
        self.points.append(pt)

        if status != OnsetStatus.NONE:
            ev = CavitationEvent(
                event_type="CAVITATION_EVENT",
                excitation_id=f"{self.sweep_id}:{self._idx}",
                frequency_hz=self.cfg.frequency_hz,
                drive_command=drive,
                measured_pressure=measured_pressure,
                onset_confidence=conf,
                acoustic_signature={
                    "rms": acoustic_rms,
                    "transient": acoustic_transient,
                    "sigma": evidence.get("acoustic_sigma"),
                },
                photon_signature={
                    "total": optical_total,
                    "peak": optical_peak,
                    "centroid": optical_centroid,
                    "latency": optical_latency,
                    "sigma": evidence.get("optical_sigma"),
                },
                timestamp=pt.timestamp,
                sweep_id=self.sweep_id,
                point_index=self._idx,
                onset_status=status.value,
            )
            self.events.append(ev)

        if status == OnsetStatus.CONFIRMED and not self._onset_found:
            self._onset_found = True
            if self.cfg.characterize_after_onset:
                self._characterize_remaining = self.cfg.characterize_steps
            else:
                self.state = CavitationState.COMPLETE
                return

        if self._onset_found and self.cfg.characterize_after_onset:
            onset_d = self.detector.onset_drive or drive
            if drive > onset_d + self.cfg.max_drive_above_onset:
                self.state = CavitationState.COMPLETE
                return
            self._characterize_remaining -= 1

        self.state = CavitationState.NEXT_STEP
        self._idx += 1

    def _summary(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "state": self.state.value,
            "fault_reason": self._fault_reason,
            "config": asdict(self.cfg),
            "n_points": len(self.points),
            "n_events": len(self.events),
            "onset_confirmed": self._onset_found,
            "onset_drive": self.detector.onset_drive,
            "points": [p.to_dict() for p in self.points],
            "events": [e.to_dict() for e in self.events],
            "baseline": {
                "acoustic_mean": self.detector.baseline_acoustic_mean,
                "acoustic_std": self.detector.baseline_acoustic_std,
                "optical_mean": self.detector.baseline_optical_mean,
                "optical_std": self.detector.baseline_optical_std,
            },
        }


def run_simulated_sweep(
    config: Optional[CavitationSweepConfig] = None,
    **sensor_kwargs: Any,
) -> Dict[str, Any]:
    """Hardware-free deterministic sweep (default path)."""
    cfg = config or CavitationSweepConfig()
    sensor = SimulatedSensorSource(seed=cfg.rng_seed or 42, **sensor_kwargs)
    sweep = CavitationSweep(config=cfg, sensor=sensor, hardware=False)
    return sweep.run()


def format_response_curve(summary: Dict[str, Any]) -> str:
    """Human-readable drive → acoustic → optical → confidence table."""
    lines = [
        f"CAVITATION.SWEEP  id={summary.get('sweep_id', '?')[:8]}  state={summary.get('state')}",
        f"onset_confirmed={summary.get('onset_confirmed')}  onset_drive={summary.get('onset_drive')}",
        "",
        f"{'idx':>4}  {'drive':>7}  {'ac_rms':>8}  {'opt_tot':>8}  {'conf':>6}  status",
        "-" * 56,
    ]
    for p in summary.get("points", []):
        lines.append(
            f"{p['point_index']:4d}  {p['drive_command']:7.3f}  "
            f"{(p.get('acoustic_rms') or 0):8.4f}  {(p.get('optical_total') or 0):8.4f}  "
            f"{p.get('onset_confidence', 0):6.3f}  {p.get('onset_status', '')}"
        )
    if summary.get("fault_reason"):
        lines.append(f"\nFAULT/ABORT reason: {summary['fault_reason']}")
    return "\n".join(lines)
