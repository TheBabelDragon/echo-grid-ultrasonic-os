#!/usr/bin/env python3
"""
Echo Grid — complete real-only visual stack

  [ Phase φ + tracks ]  [ CSI spatial map ]
  [ Actuator Δf       ]  [ Motion history + CSI bars ]

  python visualization/dashboard.py --csi
"""

from __future__ import annotations

import argparse
import sys
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
from matplotlib.patches import Circle

from echo_grid.core import EchoGridOS

TRACK_COLORS = ["#ff4d6d", "#4cc9f0", "#f4a261", "#a0e8af", "#c77dff", "#ffe066"]


class LiveDashboard:
    def __init__(self, size=16, body_port=None, csi_port=None, drive=False, closed_loop=True):
        if csi_port is None and body_port is None:
            raise SystemExit("Need --csi and/or --body")

        self.osys = EchoGridOS(size=size, body_port=body_port, csi_port=csi_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.size = size

        self.fig = plt.figure(figsize=(12, 9))
        gs = self.fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_csi_sp = self.fig.add_subplot(gs[0, 1])
        self.ax_df = self.fig.add_subplot(gs[1, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        z = np.zeros((size, size))

        # 1) Phase + tracks
        self.im_phi = self.ax_phi.imshow(
            z, cmap="viridis", vmin=-1, vmax=1, origin="lower",
            extent=[0, 1, 0, 1], animated=True,
        )
        self.ax_phi.set_title("1 · Phase field φ + tracks", fontsize=11)
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)

        self.track_rings, self.track_dots, self.track_labels = [], [], []
        for color in TRACK_COLORS:
            ring = Circle((0.5, 0.5), 0.06, fill=False, edgecolor=color, lw=2, alpha=0, zorder=6)
            dot, = self.ax_phi.plot([0.5], [0.5], "o", color=color, ms=8,
                                    markeredgecolor="w", markeredgewidth=1, alpha=0, zorder=7)
            lab = self.ax_phi.text(0.5, 0.5, "", color=color, fontsize=7, alpha=0, zorder=8)
            self.ax_phi.add_patch(ring)
            self.track_rings.append(ring)
            self.track_dots.append(dot)
            self.track_labels.append(lab)

        # 2) CSI spatial (always fills when packets arrive)
        self.im_csi = self.ax_csi_sp.imshow(
            z, cmap="magma", vmin=0, vmax=1, origin="lower",
            extent=[0, 1, 0, 1], animated=True,
        )
        self.ax_csi_sp.set_title("2 · CSI spatial map (RF residual)", fontsize=11)
        self.fig.colorbar(self.im_csi, ax=self.ax_csi_sp, fraction=0.046, pad=0.04)
        self.csi_readout = self.ax_csi_sp.text(
            0.02, 0.98, "waiting for CSI…", transform=self.ax_csi_sp.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        # 3) Actuator Δf
        self.im_df = self.ax_df.imshow(
            z, cmap="coolwarm", vmin=-500, vmax=500, origin="lower",
            extent=[0, 1, 0, 1], animated=True,
        )
        self.ax_df.set_title("3 · Actuator Δf  (40 kHz + k·φ)", fontsize=11)
        self.cbar_df = self.fig.colorbar(self.im_df, ax=self.ax_df, fraction=0.046, pad=0.04)
        self.cbar_df.set_label("Hz")
        self.df_readout = self.ax_df.text(
            0.02, 0.98, "Δf max —", transform=self.ax_df.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        # 4) Motion history + subcarriers
        self.ax_hist.set_title("4 · Motion history + subcarriers", fontsize=11)
        self.ax_hist.set_xlim(0, 200)
        self.ax_hist.set_ylim(0, 1.05)
        self.ax_hist.set_xlabel("recent packets")
        self.ax_hist.set_ylabel("motion")
        self.hist_line, = self.ax_hist.plot([], [], color="#00d4aa", lw=2.0)
        self.ax_hist.axhline(0.3, color="#555", ls="--", lw=0.8)
        self.ax_hist.axhline(0.7, color="#555", ls=":", lw=0.8)

        # small subcarrier inset axes
        self.ax_bars = self.ax_hist.inset_axes([0.55, 0.55, 0.42, 0.4])
        self.ax_bars.set_xlim(-0.5, 16.5)
        self.ax_bars.set_ylim(0, 1)
        self.ax_bars.set_xticks([])
        self.ax_bars.set_yticks([])
        self.ax_bars.set_title("CSI", fontsize=8)
        self.bars = self.ax_bars.bar(np.arange(16), np.zeros(16), color="#4cc9f0", width=0.8)

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        self.fig.suptitle(
            f"Echo Grid  ·  complete real stack  ·  {'+'.join(bits)}",
            fontsize=13,
        )
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9, family="monospace")
        self._ani = None

    def _sub_bars(self):
        pkt = self.osys.csi.last_packet if self.osys.csi else None
        if not pkt:
            return np.zeros(16)
        try:
            vals = np.array([float(x) for x in (pkt.get("csi") or [])], dtype=float)
        except (TypeError, ValueError):
            return np.zeros(16)
        if len(vals) == 0:
            return np.zeros(16)
        idx = np.linspace(0, len(vals) - 1, 16)
        return np.clip(np.interp(idx, np.arange(len(vals)), vals), 0, 1)

    def update(self, _frame):
        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)

        # 1 phase
        self.im_phi.set_array(phi)
        lo, hi = np.percentile(phi, [5, 95])
        span = max(0.12, float(hi - lo) * 0.55 + 1e-6)
        mid = float(phi.mean())
        self.im_phi.set_clim(mid - 2.5 * span, mid + 2.5 * span)

        tracks = self.osys.csi.active_tracks() if self.osys.csi else []
        for i in range(len(TRACK_COLORS)):
            ring, dot, lab = self.track_rings[i], self.track_dots[i], self.track_labels[i]
            if i < len(tracks):
                tr = tracks[i]
                x, y = tr.pos
                a = min(1.0, 0.35 + tr.confidence * 0.65)
                ring.center = (x, y)
                ring.set_radius(0.03 + 0.12 * tr.energy)
                ring.set_alpha(a)
                ring.set_linewidth(1.5 + 2.0 * tr.confidence)
                dot.set_data([x], [y])
                dot.set_alpha(a)
                lab.set_position((x + 0.03, y + 0.03))
                lab.set_text(f"{tr.track_id} {tr.state}\n{tr.confidence:.2f}")
                lab.set_alpha(a)
            else:
                ring.set_alpha(0); dot.set_alpha(0); lab.set_alpha(0)

        # 2 CSI spatial — independent of φ so it always shows when CSI is live
        if self.osys.csi is not None:
            spatial = self.osys.csi.spatial_map(self.size)
        else:
            spatial = np.zeros((self.size, self.size), dtype=np.float32)
        self.im_csi.set_array(spatial)
        smax = float(spatial.max()) + 1e-9
        self.im_csi.set_clim(0, max(0.2, smax))
        e = float(self.osys.last_csi_energy)
        self.csi_readout.set_text(
            f"motion {e:.2f}   pkts {self.osys.csi_packets}\n"
            f"tracks {len(tracks)}   rssi {self.osys.csi.last_rssi if self.osys.csi else 0:.0f}"
        )

        # 3 actuator Δf from field
        df = self.osys.mapper.delta_f(phi)
        self.im_df.set_array(df)
        abs_df = np.abs(df)
        p95 = float(np.percentile(abs_df, 95))
        fclim = max(60.0, min(2500.0, p95 * 2.0 + 30.0))
        # if field still quiet but CSI live, bias clim from motion so user sees linkage
        if abs_df.max() < 40 and e > 0.1:
            fclim = max(fclim, 200.0)
        self.im_df.set_clim(-fclim, fclim)
        self.df_readout.set_text(
            f"Δf max {float(abs_df.max()):.0f} Hz\n"
            f"scale ±{fclim:.0f} Hz\nentropy {self.osys.field.entropy:.3f}"
        )

        # 4 history + bars
        hist = self.osys.csi.motion_history if self.osys.csi else []
        if hist:
            ys = np.array(hist[-200:], dtype=float)
            xs = np.arange(len(ys))
            self.hist_line.set_data(xs, ys)
            self.ax_hist.set_xlim(0, max(50, len(ys)))
        for rect, h in zip(self.bars, self._sub_bars()):
            rect.set_height(float(h))

        self.status.set_text(
            f"motion={e:.3f}  tracks={len(tracks)}  "
            f"Δf_max={float(abs_df.max()):.0f}Hz  "
            f"csi_peak={smax:.2f}  pkts={self.osys.csi_packets}  t={self.osys.t:.1f}s"
        )
        return []

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend")
            self.osys.close()
            return
        self._ani = FuncAnimation(
            self.fig, self.update, interval=40, blit=False, cache_frame_data=False,
        )
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
    print("[dashboard] complete 4-panel real stack")
    print("  1 phase+φ tracks   2 CSI spatial   3 actuator Δf   4 motion history")
    LiveDashboard(a.size, a.body, a.csi, a.drive, not a.no_loop).run()


if __name__ == "__main__":
    main()
