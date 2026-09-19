"""
CAVITATION.SWEEP — measurement-first cavitation calibration subsystem.

Public surface re-exports. Implementation split across:
  - cavitation_sensors.py  (config, records, sensors, simulation)
  - cavitation_sweep.py    (detector, state machine, runner)
"""

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
from .cavitation_sweep import (
    CavitationDetector,
    CavitationSweep,
    run_simulated_sweep,
    format_response_curve,
)

__all__ = [
    "CavitationState",
    "OnsetStatus",
    "CavitationSweepConfig",
    "CavitationSweepPoint",
    "CavitationEvent",
    "SensorReading",
    "SensorSource",
    "SimulatedSensorSource",
    "CavitationDetector",
    "CavitationSweep",
    "run_simulated_sweep",
    "format_response_curve",
]
