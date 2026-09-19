"""
Focused tests for CAVITATION.SWEEP.

Covers:
1. clean baseline
2. monotonic response
3. false optical spike without acoustic evidence
4. acoustic anomaly without optical evidence
5. repeatable onset
6. sensor failure
7. temperature-limit abort
8. maximum-drive interlock
9. manual abort
10. complete sweep serialization
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echo_grid.cavitation import (
    CavitationDetector,
    CavitationEvent,
    CavitationState,
    CavitationSweep,
    CavitationSweepConfig,
    CavitationSweepPoint,
    OnsetStatus,
    SimulatedSensorSource,
    format_response_curve,
    run_simulated_sweep,
)


def _cfg(**kwargs) -> CavitationSweepConfig:
    defaults = dict(
        drive_start=0.0,
        drive_stop=0.40,
        drive_step=0.04,
        baseline_samples=16,
        min_repeat_count=2,
        acoustic_threshold_sigma=5.0,
        optical_threshold_sigma=4.5,
        max_drive_command=0.50,
        max_burst_duration_s=0.2,
        max_experiment_duration_s=60.0,
        max_temperature_c=45.0,
        characterize_after_onset=True,
        characterize_steps=2,
        max_drive_above_onset=0.08,
        rng_seed=42,
        settle_time_s=0.01,
        measure_window_s=0.02,
        burst_duration_s=0.02,
    )
    defaults.update(kwargs)
    return CavitationSweepConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Clean baseline
# ---------------------------------------------------------------------------

def test_clean_baseline():
    cfg = _cfg(drive_stop=0.05, drive_step=0.05)
    sensor = SimulatedSensorSource(seed=1, onset_drive=0.30, noise_level=0.01)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    assert summary["state"] == CavitationState.COMPLETE.value
    assert summary["baseline"]["acoustic_std"] >= cfg.baseline_min_std_floor
    assert summary["baseline"]["optical_std"] >= cfg.baseline_min_std_floor
    assert summary["onset_confirmed"] is False
    for p in summary["points"]:
        assert p["onset_status"] == OnsetStatus.NONE.value


def test_monotonic_response():
    cfg = _cfg(drive_stop=0.42, drive_step=0.03)
    sensor = SimulatedSensorSource(seed=7, onset_drive=0.20, noise_level=0.008)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    rms = [p["acoustic_rms"] or 0.0 for p in summary["points"]]
    mid = len(rms) // 2
    assert np.mean(rms[mid:]) > np.mean(rms[: max(1, mid // 2)])
    assert summary["n_points"] >= 5


def test_false_optical_spike_no_onset():
    cfg = _cfg(drive_stop=0.20, drive_step=0.02, min_repeat_count=2)
    sensor = SimulatedSensorSource(
        seed=3,
        onset_drive=0.35,
        false_optical_spike=True,
        noise_level=0.01,
    )
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    optical_saw_spike = any((p.get("optical_total") or 0) > 0.3 for p in summary["points"])
    assert optical_saw_spike, "fixture should inject optical spike"
    assert summary["onset_confirmed"] is False
    confirmed = [p for p in summary["points"] if p["onset_status"] == OnsetStatus.CONFIRMED.value]
    assert len(confirmed) == 0


def test_acoustic_anomaly_without_optical():
    cfg = _cfg(
        drive_stop=0.30,
        drive_step=0.03,
        require_acoustic_for_onset=True,
        require_optical_for_onset=False,
        min_repeat_count=2,
        acoustic_threshold_sigma=4.0,
    )
    sensor = SimulatedSensorSource(
        seed=5,
        onset_drive=0.40,
        acoustic_only_anomaly=True,
        noise_level=0.01,
    )
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    for p in summary["points"]:
        if p["onset_status"] != OnsetStatus.NONE.value:
            assert (p.get("acoustic_rms") or 0) > summary["baseline"]["acoustic_mean"]


def test_repeatable_onset():
    cfg = _cfg(
        drive_stop=0.40,
        drive_step=0.025,
        min_repeat_count=2,
        acoustic_threshold_sigma=5.0,
        characterize_steps=2,
    )
    sensor = SimulatedSensorSource(seed=11, onset_drive=0.18, noise_level=0.008)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    assert summary["onset_confirmed"] is True
    assert summary["onset_drive"] is not None
    assert summary["onset_drive"] >= 0.15
    statuses = [p["onset_status"] for p in summary["points"]]
    assert OnsetStatus.CANDIDATE.value in statuses or OnsetStatus.CONFIRMED.value in statuses
    assert OnsetStatus.CONFIRMED.value in statuses
    assert len(summary["events"]) >= 1
    for ev in summary["events"]:
        assert ev["event_type"] == "CAVITATION_EVENT"
        assert "acoustic_signature" in ev
        assert "photon_signature" in ev


def test_sensor_failure():
    cfg = _cfg(drive_stop=0.30, drive_step=0.05, baseline_samples=8)
    sensor = SimulatedSensorSource(seed=2, sensor_fail_after=10)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    assert summary["state"] == CavitationState.FAULT.value
    assert summary["fault_reason"] in ("simulated_sensor_loss", "sensor_loss")


def test_temperature_limit_abort():
    cfg = _cfg(
        drive_stop=0.40,
        drive_step=0.04,
        max_temperature_c=25.0,
        baseline_samples=8,
    )
    sensor = SimulatedSensorSource(
        seed=4,
        temperature_ramp=True,
        base_temperature=22.0,
        onset_drive=0.50,
    )
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    assert summary["state"] == CavitationState.FAULT.value
    assert summary["fault_reason"] == "max_temperature"


def test_maximum_drive_interlock():
    with pytest.raises(ValueError):
        CavitationSweepConfig(drive_stop=0.80, max_drive_command=0.50).validate()

    cfg = CavitationSweepConfig(
        drive_start=0.0,
        drive_stop=0.28,
        drive_step=0.04,
        max_drive_command=0.30,
        baseline_samples=8,
        settle_time_s=0.01,
        measure_window_s=0.02,
        burst_duration_s=0.02,
        rng_seed=1,
    )
    sensor = SimulatedSensorSource(seed=1, onset_drive=0.50)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    for p in summary["points"]:
        assert p["drive_command"] <= cfg.max_drive_command + 1e-9


def test_manual_abort():
    cfg = _cfg(drive_stop=0.45, drive_step=0.02, baseline_samples=8)
    sensor = SimulatedSensorSource(seed=9, onset_drive=0.40)
    sweep = CavitationSweep(config=cfg, sensor=sensor)

    original = sweep._run_one_step

    def _abort_first():
        sweep.request_abort()
        return original()

    sweep._run_one_step = _abort_first  # type: ignore
    summary = sweep.run()
    assert summary["state"] in (CavitationState.ABORT.value, CavitationState.FAULT.value)
    assert summary["fault_reason"] == "manual_abort"


def test_complete_sweep_serialization():
    cfg = _cfg(drive_stop=0.35, drive_step=0.03)
    summary = run_simulated_sweep(config=cfg)
    blob = json.dumps(summary, default=str)
    restored = json.loads(blob)
    assert restored["sweep_id"] == summary["sweep_id"]
    assert "points" in restored
    assert "events" in restored
    assert "baseline" in restored
    text = format_response_curve(summary)
    assert "CAVITATION.SWEEP" in text
    assert "drive" in text

    if summary["points"]:
        pt = CavitationSweepPoint(**{k: summary["points"][0][k] for k in summary["points"][0]})
        assert pt.to_dict()["sweep_id"] == summary["sweep_id"]
    if summary["events"]:
        ev = CavitationEvent(**{k: summary["events"][0][k] for k in summary["events"][0]})
        assert "CAVITATION_EVENT" in ev.to_json()


def test_metafield_regions():
    cfg = _cfg(drive_stop=0.35, drive_step=0.04)
    sensor = SimulatedSensorSource(seed=8, onset_drive=0.18)
    sweep = CavitationSweep(config=cfg, sensor=sensor)
    summary = sweep.run()
    assert sweep.points
    regions = sweep.to_metafield_regions()
    names = {r["region"] for r in regions}
    assert "acoustic" in names or "cavitation_confidence" in names
    for r in regions:
        assert "observed" in r
        assert "confidence" in r


def test_detector_optical_alone_insufficient():
    cfg = _cfg(require_acoustic_for_onset=True)
    det = CavitationDetector(cfg)
    det.set_baseline([0.01, 0.012, 0.011], [0.01, 0.011, 0.009])
    status, conf, _ = det.evaluate(
        acoustic_rms=0.012,
        acoustic_transient=0.01,
        optical_total=1.5,
        drive_command=0.15,
    )
    assert status == OnsetStatus.NONE
    assert conf == 0.0


def test_field_kernel_untouched():
    from echo_grid.core import EchoFieldOS, UltrasonicMapper, EchoGridOS
    field = EchoFieldOS(size=8)
    phi0 = field.phi.copy()
    field.step()
    assert field.phi.shape == (8, 8)
    mapper = UltrasonicMapper()
    df = mapper.delta_f(field.phi)
    assert df.shape == (8, 8)
    assert EchoGridOS is not None
