"""
CAVITATION.SWEEP — measurement-first cavitation calibration subsystem.

Isolated from the field kernel. Characterizes the relationship between an
abstract ultrasonic excitation command and measured acoustic / optical
response without assuming that ESP/PWM drive level equals acoustic pressure.

Safety model: fail-closed. All operating limits are explicit configuration
values supplied by the experiment operator / device calibration. The software
does not prescribe physical acoustic power limits.
"""

from __future__ import annotations

import enum
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class CavitationState(enum.Enum):
    IDLE = "IDLE"
    BASELINE = "BASELINE"
    DRIVE_STEP = "DRIVE_STEP"
    SETTLE = "SETTLE"
    MEASURE = "MEASURE"
    CLASSIFY = "CLASSIFY"
    NEXT_STEP = "NEXT_STEP"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"
    FAULT = "FAULT"


class OnsetStatus(enum.Enum):
    NONE = "NONE"
    CANDIDATE = "CAVITATION_ONSET_CANDIDATE"
    CONFIRMED = "CAVITATION_ONSET_CONFIRMED"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CavitationSweepConfig:
    """
    Explicit operator-supplied limits and sweep parameters.

    drive_command is an abstract dimensionless parameter in [0, 1].
    It is NOT assumed to equal acoustic pressure or PWM duty.
    Physical pressure calibration is supplied externally.
    """
    # Sweep grid
    frequency_hz: float = 40000.0
    drive_start: float = 0.0
    drive_stop: float = 0.45
    drive_step: float = 0.03
    burst_duration_s: float = 0.050
    settle_time_s: float = 0.080
    measure_window_s: float = 0.120

    # Baseline
    baseline_samples: int = 32
    baseline_min_std_floor: float = 1e-6

    # Onset detector (conservative, multi-evidence)
    acoustic_threshold_sigma: float = 6.0
    optical_threshold_sigma: float = 5.0
    min_repeat_count: int = 2
    require_acoustic_for_onset: bool = True
    require_optical_for_onset: bool = False

    # Bounded characterization after candidate
    characterize_after_onset: bool = True
    characterize_steps: int = 3
    max_drive_above_onset: float = 0.06

    # Hard safety interlocks (fail-closed)
    max_drive_command: float = 0.50
    max_burst_duration_s: float = 0.200
    max_experiment_duration_s: float = 120.0
    max_temperature_c: float = 45.0
    sensor_timeout_s: float = 2.0

    # Simulation / determinism
    rng_seed: Optional[int] = 42

    def validate(self) -> None:
        if self.drive_start < 0.0 or self.drive_stop < self.drive_start:
            raise ValueError("invalid drive range")
        if self.drive_step <= 0.0:
            raise ValueError("drive_step must be > 0")
        if self.burst_duration_s > self.max_burst_duration_s:
            raise ValueError("burst_duration exceeds max_burst_duration_s")
        if self.drive_stop > self.max_drive_command:
            raise ValueError("drive_stop exceeds max_drive_command interlock")
        if self.min_repeat_count < 1:
            raise ValueError("min_repeat_count must be >= 1")


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class CavitationSweepPoint:
    sweep_id: str
    point_index: int
    frequency_hz: float
    drive_command: float
    burst_duration: float
    timestamp: str
    measured_pressure: Optional[float] = None
    acoustic_rms: Optional[float] = None
    acoustic_transient: Optional[float] = None
    optical_total: Optional[float] = None
    optical_peak: Optional[float] = None
    optical_centroid: Optional[float] = None
    optical_latency: Optional[float] = None
    temperature: Optional[float] = None
    baseline_deviation: Optional[float] = None
    onset_status: str = OnsetStatus.NONE.value
    onset_confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CavitationEvent:
    """
    Serializable physical measurement event.
    No biological interpretation is present in this schema.
    """
    event_type: str
    excitation_id: str
    frequency_hz: float
    drive_command: float
    measured_pressure: Optional[float]
    onset_confidence: float
    acoustic_signature: Dict[str, Any]
    photon_signature: Dict[str, Any]
    timestamp: str
    sweep_id: str = ""
    point_index: int = -1
    onset_status: str = OnsetStatus.NONE.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Sensor abstractions (hardware boundary)
# ---------------------------------------------------------------------------

class SensorReading:
    """Normalized multi-modal reading returned by any sensor backend."""
    __slots__ = (
        "timestamp", "acoustic_rms", "acoustic_transient", "measured_pressure",
        "optical_total", "optical_peak", "optical_centroid", "optical_latency",
        "temperature", "valid", "fault",
    )

    def __init__(
        self,
        timestamp: float,
        acoustic_rms: Optional[float] = None,
        acoustic_transient: Optional[float] = None,
        measured_pressure: Optional[float] = None,
        optical_total: Optional[float] = None,
        optical_peak: Optional[float] = None,
        optical_centroid: Optional[float] = None,
        optical_latency: Optional[float] = None,
        temperature: Optional[float] = None,
        valid: bool = True,
        fault: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.acoustic_rms = acoustic_rms
        self.acoustic_transient = acoustic_transient
        self.measured_pressure = measured_pressure
        self.optical_total = optical_total
        self.optical_peak = optical_peak
        self.optical_centroid = optical_centroid
        self.optical_latency = optical_latency
        self.temperature = temperature
        self.valid = valid
        self.fault = fault


class SensorSource:
    """Abstract sensor / drive interface. Hardware implementations plug in here."""

    def set_drive(self, drive_command: float, frequency_hz: float, burst_duration_s: float) -> None:
        raise NotImplementedError

    def zero_drive(self) -> None:
        raise NotImplementedError

    def read(self) -> SensorReading:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimulatedSensorSource(SensorSource):
    """
    Deterministic simulated sensor for hardware-free CAVITATION.SWEEP.
    """

    def __init__(
        self,
        seed: int = 42,
        onset_drive: float = 0.22,
        acoustic_gain: float = 1.0,
        optical_gain: float = 1.0,
        noise_level: float = 0.015,
        false_optical_spike: bool = False,
        acoustic_only_anomaly: bool = False,
        sensor_fail_after: Optional[int] = None,
        temperature_ramp: bool = False,
        base_temperature: float = 22.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.onset_drive = float(onset_drive)
        self.acoustic_gain = float(acoustic_gain)
        self.optical_gain = float(optical_gain)
        self.noise_level = float(noise_level)
        self.false_optical_spike = bool(false_optical_spike)
        self.acoustic_only_anomaly = bool(acoustic_only_anomaly)
        self.sensor_fail_after = sensor_fail_after
        self.temperature_ramp = bool(temperature_ramp)
        self.base_temperature = float(base_temperature)

        self._drive = 0.0
        self._freq = 40000.0
        self._burst = 0.0
        self._read_count = 0
        self._t0 = time.monotonic()
        self._last_set = 0.0

    def set_drive(self, drive_command: float, frequency_hz: float, burst_duration_s: float) -> None:
        self._drive = float(np.clip(drive_command, 0.0, 1.0))
        self._freq = float(frequency_hz)
        self._burst = float(max(0.0, burst_duration_s))
        self._last_set = time.monotonic()

    def zero_drive(self) -> None:
        self._drive = 0.0
        self._burst = 0.0

    def read(self) -> SensorReading:
        self._read_count += 1
        now = time.monotonic()

        if self.sensor_fail_after is not None and self._read_count > self.sensor_fail_after:
            return SensorReading(
                timestamp=now,
                valid=False,
                fault="simulated_sensor_loss",
            )

        x = max(0.0, self._drive - self.onset_drive)
        acoustic_base = self.acoustic_gain * (x ** 1.4) * 2.5
        acoustic_rms = acoustic_base + float(self.rng.normal(0.0, self.noise_level))
        acoustic_transient = 0.0
        if self._drive > self.onset_drive * 0.95:
            acoustic_transient = self.acoustic_gain * (x ** 0.8) * 1.8 + float(
                self.rng.normal(0.0, self.noise_level * 1.5)
            )

        if self.acoustic_only_anomaly and self._drive > 0.15:
            acoustic_rms += 0.35
            acoustic_transient += 0.5

        optical_total = 0.0
        optical_peak = 0.0
        optical_centroid = 0.5
        optical_latency = None
        if acoustic_base > 0.02 and not self.acoustic_only_anomaly:
            optical_total = self.optical_gain * acoustic_base * 0.9 + float(
                self.rng.normal(0.0, self.noise_level * 0.8)
            )
            optical_peak = optical_total * (1.2 + 0.3 * self.rng.random())
            optical_centroid = 0.35 + 0.3 * min(1.0, acoustic_base)
            optical_latency = 0.008 + 0.004 * self.rng.random()

        if self.false_optical_spike and 0.10 < self._drive < 0.18:
            optical_total = 0.55 + float(self.rng.normal(0.0, 0.02))
            optical_peak = optical_total * 1.4
            optical_centroid = 0.6

        measured_pressure = None
        if acoustic_rms is not None:
            measured_pressure = max(0.0, acoustic_rms * 12.0)

        temp = self.base_temperature
        if self.temperature_ramp:
            temp += 0.15 * self._read_count + 3.0 * self._drive

        return SensorReading(
            timestamp=now,
            acoustic_rms=max(0.0, acoustic_rms),
            acoustic_transient=max(0.0, acoustic_transient),
            measured_pressure=measured_pressure,
            optical_total=max(0.0, optical_total),
            optical_peak=max(0.0, optical_peak),
            optical_centroid=optical_centroid,
            optical_latency=optical_latency,
            temperature=temp,
            valid=True,
            fault=None,
        )

    def close(self) -> None:
        self.zero_drive()
