#!/usr/bin/env python3
"""
Echo Grid live visualization

  python visualization/dashboard.py --csi
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

def _configure_backend() -> str:
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue
    matplotlib.use("Agg", force=True)
    return "Agg"

_BACKEND = _configure_backend()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from echo_grid.core import EchoGridOS


class LiveDashboard:
    def __init__(
        self,
        size: int = 16,
        body_port=None,
        csi_port=None,
        drive: bool = False,
        closed_loop: bool = True,
        demo: bool = True,
    ):
        self.osys = EchoGridOS(size=size, body_port=body_port, csi_port=csi_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.demo = demo

        self.fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        self.ax_phi, self.ax_freq = axes

        self.im_phi = self.ax_phi.imshow(
            np.zeros((size, size)), cmap="viridis",
            vmin=-2.0, vmax=2.0, animated=True, origin="lower",
        )
        self.im_freq = self.ax_freq.imshow(
            np.zeros((size, size)), cmap="plasma",
            vmin=38000, vmax=42000, animated=True, origin="lower",
        )

        self.ax_phi.set_title("Phase field φ")
        self.ax_freq.set_title("Frequency map (Hz)")
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)
        self.fig.colorbar(self.im_freq, ax=self.ax_freq, fraction=0.046, pad=0.04)

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        if not bits:
            bits.append("soft")
        self.fig.suptitle(f"Echo Grid  ·  mode={'+'.join(bits)}", fontsize=13)

        self.status = self.fig.text(
            0.5, 0.02,
            "entropy=0  obs=0  csi=0  pkts=0  t=0",
            ha="center", fontsize=10, family="monospace",
        )
        self._t0 = time.time()
        self._ani = None

    def update(self, _frame):
        if self.demo:
            t = time.time() - self._t0
            self.osys.touch(
                0.5 + 0.30 * np.sin(t * 0.55),
                0.5 + 0.30 * np.cos(t * 0.41),
                strength=0.25,
            )

        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)
        freq = 40000.0 + phi * 2000.0

        self.im_phi.set_array(phi)
        self.im_freq.set_array(freq)
        self.status.set_text(
            f"entropy={self.osys.field.entropy:.3f}   "
            f"obs={self.osys.last_obs:.3f}   "
            f"csi={self.osys.last_csi_energy:.3f}   "
            f"pkts={self.osys.csi_packets}   "
            f"t={self.osys.t:.1f}s"
        )
        return [self.im_phi, self.im_freq, self.status]

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend — install tk: sudo pacman -S tk")
            self.osys.close()
            return

        self._ani = FuncAnimation(
            self.fig, self.update, interval=40, blit=False, cache_frame_data=False,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        try:
            plt.show()
        finally:
            self.osys.save(force=True)
            self.osys.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", nargs="?", const="", default=None)
    parser.add_argument("--csi", nargs="?", const=4210, type=int, default=None)
    parser.add_argument("--drive", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-demo", action="store_true")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()

    print(f"[dashboard] backend={_BACKEND}  csi_port={args.csi}")

    LiveDashboard(
        size=args.size,
        body_port=args.body,
        csi_port=args.csi,
        drive=args.drive,
        closed_loop=not args.no_loop,
        demo=not args.no_demo,
    ).run()


if __name__ == "__main__":
    main()
