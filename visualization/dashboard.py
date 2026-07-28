#!/usr/bin/env python3
"""
Echo Grid — Live Field Visualization

  python visualization/dashboard.py
  python visualization/dashboard.py --body
  python visualization/dashboard.py --body --drive
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from echo_grid.core import EchoGridOS


class LiveDashboard:
    def __init__(self, size: int = 16, body_port=None, drive: bool = False, closed_loop: bool = True):
        self.osys = EchoGridOS(size=size, body_port=body_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.size = size

        self.fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        self.ax_phi, self.ax_freq = axes

        self.im_phi = self.ax_phi.imshow(
            np.zeros((size, size)),
            cmap="viridis",
            vmin=-2.0,
            vmax=2.0,
            animated=True,
            origin="lower",
        )
        self.im_freq = self.ax_freq.imshow(
            np.zeros((size, size)),
            cmap="plasma",
            vmin=38000,
            vmax=42000,
            animated=True,
            origin="lower",
        )

        self.ax_phi.set_title("Phase field φ")
        self.ax_freq.set_title("Frequency map (Hz)")
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)
        self.fig.colorbar(self.im_freq, ax=self.ax_freq, fraction=0.046, pad=0.04)

        mode = "body" if self.osys.body_connected else "soft"
        self.fig.suptitle(f"Echo Grid Ultrasonic OS  ·  mode={mode}", fontsize=13)

        self.status = self.fig.text(
            0.5, 0.02,
            "entropy=0.000   obs=0.000   t=0.0s",
            ha="center", fontsize=10, family="monospace",
        )

        self._t0 = time.time()

    def _excite(self):
        t = time.time() - self._t0
        x = 0.5 + 0.30 * np.sin(t * 0.55)
        y = 0.5 + 0.30 * np.cos(t * 0.41)
        self.osys.touch(x, y, strength=0.55)

    def update(self, _frame):
        self._excite()
        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)

        freq = 40000.0 + phi * 2000.0

        self.im_phi.set_array(phi)
        self.im_freq.set_array(freq)

        self.status.set_text(
            f"entropy={self.osys.field.entropy:.3f}   "
            f"obs={self.osys.last_obs:.3f}   "
            f"t={self.osys.t:.1f}s"
        )
        return [self.im_phi, self.im_freq, self.status]

    def run(self):
        ani = FuncAnimation(
            self.fig,
            self.update,
            interval=40,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        try:
            plt.show()
        finally:
            self.osys.save(force=True)
            self.osys.close()


def main():
    parser = argparse.ArgumentParser(description="Echo Grid live visualization")
    parser.add_argument("--body", nargs="?", const="", default=None,
                        help="Attach Echo Body (optional serial port)")
    parser.add_argument("--drive", action="store_true",
                        help="Drive physical emitters from field regions")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable closed-loop feedback")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()

    dash = LiveDashboard(
        size=args.size,
        body_port=args.body,
        drive=args.drive,
        closed_loop=not args.no_loop,
    )
    dash.run()


if __name__ == "__main__":
    main()
