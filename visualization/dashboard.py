#!/usr/bin/env python3
"""
Echo Grid live visualization — field + CSI observation panel

  python visualization/dashboard.py --csi
  python visualization/dashboard.py --csi --no-demo
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
from matplotlib.patches import Circle, FancyBboxPatch

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
        self.size = size

        self.fig = plt.figure(figsize=(13, 5.5))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.7], wspace=0.28)
        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_freq = self.fig.add_subplot(gs[0, 1])
        self.ax_csi = self.fig.add_subplot(gs[0, 2])

        # --- phase ---
        self.im_phi = self.ax_phi.imshow(
            np.zeros((size, size)), cmap="viridis",
            vmin=-2.0, vmax=2.0, animated=True, origin="lower",
            extent=[0, 1, 0, 1],
        )
        self.ax_phi.set_title("Phase field φ")
        self.ax_phi.set_xlabel("x")
        self.ax_phi.set_ylabel("y")
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)

        # CSI injection marker on phase field
        self.csi_marker, = self.ax_phi.plot(
            [0.5], [0.5], "o", color="#ff4d6d", markersize=14,
            markeredgecolor="white", markeredgewidth=1.5, alpha=0.0, zorder=5,
        )
        self.csi_ring = Circle(
            (0.5, 0.5), 0.08, fill=False, edgecolor="#ff4d6d",
            linewidth=2, alpha=0.0, zorder=5,
        )
        self.ax_phi.add_patch(self.csi_ring)

        # --- frequency ---
        self.im_freq = self.ax_freq.imshow(
            np.zeros((size, size)), cmap="plasma",
            vmin=38000, vmax=42000, animated=True, origin="lower",
            extent=[0, 1, 0, 1],
        )
        self.ax_freq.set_title("Frequency map (Hz)")
        self.fig.colorbar(self.im_freq, ax=self.ax_freq, fraction=0.046, pad=0.04)

        # --- CSI observation panel ---
        self.ax_csi.set_title("CSI observation")
        self.ax_csi.set_xlim(0, 1)
        self.ax_csi.set_ylim(0, 1)
        self.ax_csi.axis("off")

        # energy bar background
        self.energy_bg = FancyBboxPatch(
            (0.12, 0.72), 0.76, 0.12,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#1a1a2e", edgecolor="#444", linewidth=1,
        )
        self.ax_csi.add_patch(self.energy_bg)
        self.energy_bar = FancyBboxPatch(
            (0.12, 0.72), 0.02, 0.12,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#00d4aa", edgecolor="none",
        )
        self.ax_csi.add_patch(self.energy_bar)
        self.energy_label = self.ax_csi.text(
            0.5, 0.88, "energy  0.00", ha="center", va="bottom",
            fontsize=11, family="monospace", color="#ddd",
        )

        self.rssi_text = self.ax_csi.text(
            0.5, 0.62, "rssi  —", ha="center", fontsize=10,
            family="monospace", color="#aaa",
        )
        self.pkts_text = self.ax_csi.text(
            0.5, 0.54, "pkts  0", ha="center", fontsize=10,
            family="monospace", color="#aaa",
        )

        # subcarrier bars
        self.n_bars = 16
        self.bar_container = self.ax_csi.bar(
            np.linspace(0.12, 0.88, self.n_bars),
            np.zeros(self.n_bars),
            width=0.04,
            bottom=0.12,
            color="#4cc9f0",
            align="center",
        )
        self.ax_csi.text(
            0.5, 0.06, "subcarriers", ha="center", fontsize=9, color="#888",
        )

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        if not bits:
            bits.append("soft")
        self.fig.suptitle(f"Echo Grid  ·  mode={'+'.join(bits)}", fontsize=13)

        self.status = self.fig.text(
            0.5, 0.01,
            "entropy=0  obs=0  csi=0  pkts=0",
            ha="center", fontsize=9, family="monospace",
        )

        self._t0 = time.time()
        self._ani = None
        self._last_xy = (0.5, 0.5)

    def _csi_bars(self):
        pkt = None
        if self.osys.csi is not None:
            pkt = self.osys.csi.last_packet
        if not pkt:
            return np.zeros(self.n_bars)

        csi = pkt.get("csi") or []
        try:
            vals = np.array([float(x) for x in csi], dtype=float)
        except (TypeError, ValueError):
            return np.zeros(self.n_bars)

        if len(vals) == 0:
            return np.zeros(self.n_bars)

        # downsample / pad to n_bars
        idx = np.linspace(0, len(vals) - 1, self.n_bars)
        sampled = np.interp(idx, np.arange(len(vals)), vals)
        return np.clip(sampled, 0, 1)

    def update(self, _frame):
        if self.demo:
            t = time.time() - self._t0
            self.osys.touch(
                0.5 + 0.28 * np.sin(t * 0.55),
                0.5 + 0.28 * np.cos(t * 0.41),
                strength=0.22,
            )

        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)
        freq = 40000.0 + phi * 2000.0

        self.im_phi.set_array(phi)
        self.im_freq.set_array(freq)

        # CSI observation → visual panel + marker
        e = float(self.osys.last_csi_energy)
        self.energy_bar.set_width(max(0.02, 0.76 * e))
        # color: teal → amber → red with energy
        if e < 0.35:
            color = "#00d4aa"
        elif e < 0.7:
            color = "#f4a261"
        else:
            color = "#e63946"
        self.energy_bar.set_facecolor(color)
        self.energy_label.set_text(f"energy  {e:.2f}")

        rssi = -90.0
        if self.osys.csi is not None:
            rssi = self.osys.csi.last_rssi
            x, y, force = self.osys.csi.injection_point()
            self._last_xy = (x, y)
        else:
            force = 0.0

        self.rssi_text.set_text(f"rssi  {rssi:.0f} dBm")
        self.pkts_text.set_text(f"pkts  {self.osys.csi_packets}")

        # injection marker on phase field
        alpha = min(1.0, e * 1.4)
        self.csi_marker.set_data([self._last_xy[0]], [self._last_xy[1]])
        self.csi_marker.set_alpha(alpha)
        self.csi_ring.center = self._last_xy
        self.csi_ring.set_radius(0.05 + 0.12 * e)
        self.csi_ring.set_alpha(alpha * 0.85)

        # subcarrier bars
        bars = self._csi_bars()
        for rect, h in zip(self.bar_container, bars):
            rect.set_height(0.32 * h)

        self.status.set_text(
            f"entropy={self.osys.field.entropy:.3f}   "
            f"obs={self.osys.last_obs:.3f}   "
            f"csi={e:.3f}   "
            f"pkts={self.osys.csi_packets}   "
            f"t={self.osys.t:.1f}s"
        )
        return []

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend — sudo pacman -S tk")
            self.osys.close()
            return

        self._ani = FuncAnimation(
            self.fig, self.update, interval=40, blit=False, cache_frame_data=False,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
