#!/usr/bin/env python3
"""
Echo Grid live view — real CSI/body only (no synthetic demo)

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
    for b in ("TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg"):
        try:
            matplotlib.use(b, force=True)
            return b
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

TRACK_COLORS = ["#ff4d6d", "#4cc9f0", "#f4a261", "#a0e8af", "#c77dff", "#ffe066"]
STATE_LS = {"idle": ":", "move": "-", "surge": "-"}


class LiveDashboard:
    def __init__(self, size=16, body_port=None, csi_port=None, drive=False, closed_loop=True):
        if csi_port is None and body_port is None:
            raise SystemExit("Need --csi and/or --body (real inputs only)")

        self.osys = EchoGridOS(size=size, body_port=body_port, csi_port=csi_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.size = size

        self.fig = plt.figure(figsize=(13, 5.5))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.7], wspace=0.28)
        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_freq = self.fig.add_subplot(gs[0, 1])
        self.ax_csi = self.fig.add_subplot(gs[0, 2])

        self.im_phi = self.ax_phi.imshow(
            np.zeros((size, size)), cmap="viridis", vmin=-2, vmax=2,
            animated=True, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_phi.set_title("Phase φ + real tracks")
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)

        self.track_rings, self.track_dots, self.track_labels = [], [], []
        for color in TRACK_COLORS:
            ring = Circle((0.5, 0.5), 0.06, fill=False, edgecolor=color, lw=2, alpha=0, zorder=6)
            dot, = self.ax_phi.plot([0.5], [0.5], "o", color=color, ms=8,
                                    markeredgecolor="white", markeredgewidth=1, alpha=0, zorder=7)
            lab = self.ax_phi.text(0.5, 0.5, "", color=color, fontsize=7, ha="left", va="bottom", alpha=0, zorder=8)
            self.ax_phi.add_patch(ring)
            self.track_rings.append(ring)
            self.track_dots.append(dot)
            self.track_labels.append(lab)

        self.im_freq = self.ax_freq.imshow(
            np.zeros((size, size)), cmap="coolwarm", vmin=-1500, vmax=1500,
            animated=True, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_freq.set_title("Frequency Δf (Hz from 40 kHz)")
        self.fig.colorbar(self.im_freq, ax=self.ax_freq, fraction=0.046, pad=0.04)

        self.ax_csi.set_title("CSI (real)")
        self.ax_csi.set_xlim(0, 1)
        self.ax_csi.set_ylim(0, 1)
        self.ax_csi.axis("off")
        self.ax_csi.add_patch(FancyBboxPatch((0.12, 0.72), 0.76, 0.12, boxstyle="round,pad=0.01,rounding_size=0.02",
                                             facecolor="#1a1a2e", edgecolor="#444"))
        self.energy_bar = FancyBboxPatch((0.12, 0.72), 0.02, 0.12, boxstyle="round,pad=0.01,rounding_size=0.02",
                                         facecolor="#00d4aa", edgecolor="none")
        self.ax_csi.add_patch(self.energy_bar)
        self.energy_label = self.ax_csi.text(0.5, 0.88, "motion  0.00", ha="center", va="bottom",
                                             fontsize=11, family="monospace", color="#ddd")
        self.rssi_text = self.ax_csi.text(0.5, 0.62, "rssi  —", ha="center", fontsize=10, family="monospace", color="#aaa")
        self.pkts_text = self.ax_csi.text(0.5, 0.54, "pkts  0", ha="center", fontsize=10, family="monospace", color="#aaa")
        self.tracks_text = self.ax_csi.text(0.5, 0.46, "tracks  0", ha="center", fontsize=10, family="monospace", color="#aaa")
        self.n_bars = 16
        self.bars = self.ax_csi.bar(np.linspace(0.12, 0.88, self.n_bars), np.zeros(self.n_bars),
                                    width=0.04, bottom=0.10, color="#4cc9f0", align="center")
        self.ax_csi.text(0.5, 0.04, "subcarriers", ha="center", fontsize=9, color="#888")

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        self.fig.suptitle(f"Echo Grid  ·  real-only  ·  {'+'.join(bits) or 'waiting'}", fontsize=12)
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9, family="monospace")
        self._ani = None

    def _bars(self):
        pkt = self.osys.csi.last_packet if self.osys.csi else None
        if not pkt:
            return np.zeros(self.n_bars)
        try:
            vals = np.array([float(x) for x in (pkt.get("csi") or [])], dtype=float)
        except (TypeError, ValueError):
            return np.zeros(self.n_bars)
        if len(vals) == 0:
            return np.zeros(self.n_bars)
        idx = np.linspace(0, len(vals) - 1, self.n_bars)
        return np.clip(np.interp(idx, np.arange(len(vals)), vals), 0, 1)

    def update(self, _frame):
        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)

        self.im_phi.set_array(phi)
        pmax = float(np.max(np.abs(phi))) + 1e-6
        clim = max(0.25, min(2.0, pmax * 1.2))
        self.im_phi.set_clim(-clim, clim)

        df = phi * 2000.0
        self.im_freq.set_array(df)
        fmax = float(np.max(np.abs(df))) + 1.0
        fclim = max(150.0, min(2000.0, fmax * 1.25))
        self.im_freq.set_clim(-fclim, fclim)

        e = float(self.osys.last_csi_energy)
        self.energy_bar.set_width(max(0.02, 0.76 * e))
        self.energy_bar.set_facecolor("#00d4aa" if e < 0.35 else ("#f4a261" if e < 0.7 else "#e63946"))
        self.energy_label.set_text(f"motion  {e:.2f}")
        rssi = self.osys.csi.last_rssi if self.osys.csi else -90
        self.rssi_text.set_text(f"rssi  {rssi:.0f} dBm")
        self.pkts_text.set_text(f"pkts  {self.osys.csi_packets}")

        tracks = self.osys.csi.active_tracks() if self.osys.csi else []
        self.tracks_text.set_text(f"tracks  {len(tracks)}")

        for i in range(len(TRACK_COLORS)):
            ring, dot, lab = self.track_rings[i], self.track_dots[i], self.track_labels[i]
            if i < len(tracks):
                tr = tracks[i]
                x, y = tr.pos
                a = min(1.0, 0.3 + tr.confidence * 0.7)
                ring.center = (x, y)
                ring.set_radius(0.03 + 0.12 * tr.energy)
                ring.set_alpha(a)
                ring.set_linestyle(STATE_LS.get(tr.state, "-"))
                ring.set_linewidth(1.5 + 2.0 * tr.confidence)
                dot.set_data([x], [y])
                dot.set_alpha(a)
                lab.set_position((x + 0.03, y + 0.03))
                lab.set_text(f"{tr.track_id} {tr.state}\n{tr.confidence:.2f}")
                lab.set_alpha(a)
            else:
                ring.set_alpha(0); dot.set_alpha(0); lab.set_alpha(0)

        for rect, h in zip(self.bars, self._bars()):
            rect.set_height(0.28 * h)

        self.status.set_text(
            f"entropy={self.osys.field.entropy:.3f}  motion={e:.3f}  "
            f"tracks={len(tracks)}  Δf=±{fclim:.0f}Hz  pkts={self.osys.csi_packets}  t={self.osys.t:.1f}s"
        )
        return []

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend"); self.osys.close(); return
        self._ani = FuncAnimation(self.fig, self.update, interval=40, blit=False, cache_frame_data=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            plt.show()
        finally:
            self.osys.save(force=True)
            self.osys.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--body", nargs="?", const="", default=None)
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=None)
    p.add_argument("--drive", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()
    print(f"[dashboard] real-only  backend={_BACKEND}  csi={a.csi}")
    LiveDashboard(a.size, a.body, a.csi, a.drive, not a.no_loop).run()


if __name__ == "__main__":
    main()
