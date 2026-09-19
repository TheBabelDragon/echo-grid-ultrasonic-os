#!/usr/bin/env python3
"""
CAVITATION.SWEEP software-only demonstration.

Default: fully simulated sensors — no ultrasonic transducer, tank, or
high-power hardware is energised.

  python tools/cavitation_sweep_demo.py
  python tools/cavitation_sweep_demo.py --drive-stop 0.40 --step 0.025
  python tools/cavitation_sweep_demo.py --json /tmp/sweep.json

Hardware path requires explicit --hardware (and a registered SensorSource);
it is not implemented in the default tree and will refuse to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_grid.cavitation import (
    CavitationSweep,
    CavitationSweepConfig,
    SimulatedSensorSource,
    format_response_curve,
    run_simulated_sweep,
)


def main() -> int:
    p = argparse.ArgumentParser(description="CAVITATION.SWEEP demo (simulation default)")
    p.add_argument("--hardware", action="store_true",
                   help="Opt-in to real hardware path (refused if no backend)")
    p.add_argument("--drive-start", type=float, default=0.0)
    p.add_argument("--drive-stop", type=float, default=0.40)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--freq", type=float, default=40000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", type=str, default=None, help="Write full summary JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.hardware:
        print("ERROR: --hardware requested but no physical SensorSource is registered.")
        print("Simulation is the only supported path in this tree.")
        print("Supply a calibrated SensorSource implementation and re-run.")
        return 2

    cfg = CavitationSweepConfig(
        frequency_hz=args.freq,
        drive_start=args.drive_start,
        drive_stop=args.drive_stop,
        drive_step=args.step,
        max_drive_command=max(args.drive_stop, 0.50),
        rng_seed=args.seed,
    )

    summary = run_simulated_sweep(config=cfg)

    if not args.quiet:
        print(format_response_curve(summary))
        print()
        if summary.get("events"):
            print(f"Events emitted: {len(summary['events'])}")
            for ev in summary["events"]:
                print(f"  [{ev['onset_status']}] drive={ev['drive_command']:.3f} "
                      f"conf={ev['onset_confidence']:.3f}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote summary → {path.resolve()}")

    return 0 if summary.get("state") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
