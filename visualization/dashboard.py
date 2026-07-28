#!/usr/bin/env python3
"""
Echo Grid live visualization — adaptive frequency map + multi-track circles

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

TRACK_COLORS = ["#ff4d6d", "#4cc9f0", "#f4a261", "#a0e8af", "#c77dff", "#ffe066"]
BASE_FREQ = 40000.0


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

        self.im_phi = self.ax_phi.imshow(
            np.zeros((size, size)), cmap="viridis",
            vmin=-2.0, vmax=2.0, animated=True, origin="lower",
            extent=[0, 1, 0, 1],
        )
        self.ax_phi.set_title("Phase field φ + tracks")
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)

        self.track_rings, self.track_dots, self.track_labels = [], [], []
        for color in TRACK_COLORS:
            ring = Circle((0.5, 0.5), 0.06, fill=False, edgecolor=color,
                          linewidth=2.2, alpha=0.0, zorder=6)
            dot, = self.ax_phi.plot([0.5], [0.5], "o", color=color, markersize=9,
                                    markeredgecolor="white", markeredgewidth=1.0,
                                    alpha=0.0, zorder=7)
            lab = self.ax_phi.text(0.5, 0.5, "", color=color, fontsize=7,
                                   ha="left", va="bottom", alpha=0.0, zorder=8)
            self.ax_phi.add_patch(ring)
            self.track_rings.append(ring)
            self.track_dots.append(dot)
            self.track_labels.append(lab)

        # Frequency shown as Δf from 40 kHz so structure is visible
        self.im_freq = self.ax_freq.imshow(
            np.zeros((size, size)), cmap="coolwarm",
            vmin=-1500, vmax=1500, animated=True, origin="lower",
            extent=[0, 1, 0, 1],
        )
        self.ax_freq.set_title("Frequency Δf (Hz from 40 kHz)")
        self.cbar_freq = self.fig.colorbar(
            self.im_freq, ax=self.ax_freq, fraction=0.046, pad=0.04
        )

        # CSI panel
        self.ax_csi.set_title("CSI observation")
        self.ax_csi.set_xlim(0, 1)
        self.ax_csi.set_ylim(0, 1)
        self.ax_csi.axis("off")

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
        self.tracks_text = self.ax_csi.text(
            0.5, 0.46, "tracks  0", ha="center", fontsize=10,
            family="monospace", color="#aaa",
        )

        self.n_bars = 16
        self.bar_container = self.ax_csi.bar(
            np.linspace(0.12, 0.88, self.n_bars),
            np.zeros(self.n_bars),
            width=0.04, bottom=0.10, color="#4cc9f0", align="center",
        )
        self.ax_csi.text(0.5, 0.04, "subcarriers", ha="center", fontsize=9, color="#888")

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        if not bits:
            bits.append("soft")
        self.fig.suptitle(f"Echo Grid  ·  mode={'+'.join(bits)}", fontsize=13)

        self.status = self.fig.text(
            0.5, 0.01, "", ha="center", fontsize=9, family="monospace",
        )
        self._t0 = time.time()
        self._ani = None

    def _csi_bars(self):
        pkt = self.osys.csi.last_packet if self.osys.csi else None
        if not pkt:
            return np.zeros(self.n_bars)
        csi = pkt.get("csi") or []
        try:
            vals = np.array([float(x) for x in csi], dtype=float)
        except (TypeError, ValueError):
            return np.zeros(self.n_bars)
        if len(vals) == 0:
            return np.zeros(self.n_bars)
        idx = np.linspace(0, len(vals) - 1, self.n_bars)
        return np.clip(np.interp(idx, np.arange(len(vals)), vals), 0, 1)

    def update(self, _frame):
        if self.demo:
            t = time.time() - self._t0
            self.osys.touch(
                0.5 + 0.28 * np.sin(t * 0.55),
                0.5 + 0.28 * np.cos(t * 0.41),
                strength=0.18,
            )

        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)

        # Phase
        self.im_phi.set_array(phi)
        # Adaptive phase scale so weak fields still show structure
        pmax = float(np.max(np.abs(phi))) + 1e-6
        clim = max(0.35, min(2.0, pmax * 1.15))
        self.im_phi.set_clim(-clim, clim)

        # Frequency as deviation from base — THIS is what was looking empty before
        df = phi * 2000.0  # Hz offset from 40 kHz
        self.im_freq.set_array(df)
        fmax = float(np.max(np.abs(df))) + 1.0
        fclim = max(200.0, min(2000.0, fmax * 1.2))
        self.im_freq.set_clim(-fclim, fclim)

        e = float(self.osys.last_csi_energy)
        self.energy_bar.set_width(max(0.02, 0.76 * e))
        self.energy_bar.set_facecolor(
            "#00d4aa" if e < 0.35 else ("#f4a261" if e < 0.7 else "#e63946")
        )
        self.energy_label.set_text(f"energy  {e:.2f}")

        rssi = self.osys.csi.last_rssi if self.osys.csi else -90.0
        self.rssi_text.set_text(f"rssi  {rssi:.0f} dBm")
        self.pkts_text.set_text(f"pkts  {self.osys.csi_packets}")

        tracks = self.osys.csi.active_tracks() if self.osys.csi else []
        self.tracks_text.set_text(f"tracks  {len(tracks)}")

        for i in range(len(TRACK_COLORS)):
            ring = self.track_rings[i]
            dot = self.track_dots[i]
            lab = self.track_labels[i]
            if i < len(tracks):
                tr = tracks[i]
                alpha = min(1.0, 0.3 + tr.confidence * 0.7)
                radius = 0.035 + 0.11 * tr.energy
                ring.center = (tr.x, tr.y)
                ring.set_radius(radius)
                ring.set_alpha(alpha)
                ring.set_linewidth(1.5 + 2.0 * tr.confidence)
                dot.set_data([tr.x], [tr.y])
                dot.set_alpha(alpha)
                short = tr.track_id if len(tr.track_id) <= 12 else tr.track_id[:10] + "…"
                lab.set_position((tr.x + 0.03, tr.y + 0.03))
                lab.set_text(f"{short}\n{tr.confidence:.2f}")
                lab.set_alpha(alpha)
            else:
                ring.set_alpha(0.0)
                dot.set_alpha(0.0)
                lab.set_alpha(0.0)

        for rect, h in zip(self.bar_container, self._csi_bars()):
            rect.set_height(0.28 * h)

        self.status.set_text(
            f"entropy={self.osys.field.entropy:.3f}   "
            f"csi={e:.3f}   tracks={len(tracks)}   "
            f"Δf=±{fclim:.0f}Hz   pkts={self.osys.csi_packets}   "
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
