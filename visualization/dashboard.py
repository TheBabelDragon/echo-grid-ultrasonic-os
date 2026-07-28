#!/usr/bin/env python3
"""
Echo Grid live stack — CSI enabled by default

  python visualization/dashboard.py
  python visualization/dashboard.py --body          # + ultrasonic body
  python visualization/dashboard.py --no-csi        # software field only (no RF)
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
from matplotlib.patches import Circle

from echo_grid.core import EchoGridOS

TRACK_COLORS = ["#ff4d6d", "#4cc9f0", "#f4a261", "#a0e8af", "#c77dff", "#ffe066"]


class LiveDashboard:
    def __init__(self, size=16, body_port=None, csi_port=4210, drive=False, closed_loop=True):
        self.osys = EchoGridOS(size=size, body_port=body_port, csi_port=csi_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.size = size
        self._frame = 0
        self._last_log = 0.0
        self._running = True
        self._df_clim = 300.0

        self.fig = plt.figure(figsize=(12, 9))
        gs = self.fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_csi_sp = self.fig.add_subplot(gs[0, 1])
        self.ax_df = self.fig.add_subplot(gs[1, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        z = np.zeros((size, size), dtype=np.float32)

        self.im_phi = self.ax_phi.imshow(
            z.copy(), cmap="viridis", vmin=-1, vmax=1, origin="lower", extent=[0, 1, 0, 1],
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

        self.im_csi = self.ax_csi_sp.imshow(
            z.copy(), cmap="magma", vmin=0, vmax=1, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_csi_sp.set_title("2 · CSI spatial map", fontsize=11)
        self.fig.colorbar(self.im_csi, ax=self.ax_csi_sp, fraction=0.046, pad=0.04)
        self.csi_readout = self.ax_csi_sp.text(
            0.02, 0.98, "waiting…", transform=self.ax_csi_sp.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        self.im_df = self.ax_df.imshow(
            z.copy(), cmap="inferno", vmin=0, vmax=400, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_df.set_title("3 · Actuator |Δf| (Hz from 40 kHz)", fontsize=11)
        self.cbar_df = self.fig.colorbar(self.im_df, ax=self.ax_df, fraction=0.046, pad=0.04)
        self.cbar_df.set_label("|Hz|")
        self.df_readout = self.ax_df.text(
            0.02, 0.98, "|Δf| max —", transform=self.ax_df.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        self.ax_hist.set_title("4 · Motion history + subcarriers", fontsize=11)
        self.ax_hist.set_xlim(0, 200)
        self.ax_hist.set_ylim(0, 1.05)
        self.hist_line, = self.ax_hist.plot([], [], color="#00d4aa", lw=2.0)
        self.ax_hist.axhline(0.3, color="#555", ls="--", lw=0.8)
        self.ax_bars = self.ax_hist.inset_axes([0.55, 0.55, 0.42, 0.4])
        self.ax_bars.set_xlim(-0.5, 16.5)
        self.ax_bars.set_ylim(0, 1)
        self.ax_bars.set_xticks([])
        self.ax_bars.set_yticks([])
        self.bars = self.ax_bars.bar(np.arange(16), np.zeros(16), color="#4cc9f0", width=0.8)

        bits = []
        if self.osys.body_connected:
            bits.append("body")
        if self.osys.csi_enabled:
            bits.append("csi")
        if not bits:
            bits.append("idle")
        self.fig.suptitle(f"Echo Grid  ·  {'+'.join(bits)}", fontsize=13)
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9, family="monospace")
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "_running", False))

    def _sub_bars(self):
        if not self.osys.csi:
            return np.zeros(16)
        pkt = self.osys.csi.last_packet
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

    def tick(self):
        if not self._running:
            return
        self._frame += 1
        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)
        phi_view = np.array(phi, dtype=np.float32, copy=True)

        self.im_phi.set_data(phi_view)
        lo, hi = np.percentile(phi_view, [5, 95])
        span = max(0.12, float(hi - lo) * 0.55 + 1e-6)
        mid = float(phi_view.mean())
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

        spatial = (
            np.array(self.osys.csi.spatial_map(self.size), dtype=np.float32, copy=True)
            if self.osys.csi else np.zeros((self.size, self.size), dtype=np.float32)
        )
        self.im_csi.set_data(spatial)
        self.im_csi.set_clim(0.0, max(0.25, float(spatial.max()) + 1e-9))
        e = float(self.osys.last_csi_energy)
        self.csi_readout.set_text(
            f"motion {e:.2f}   pkts {self.osys.csi_packets}\n"
            f"tracks {len(tracks)}   rssi {self.osys.csi.last_rssi if self.osys.csi else 0:.0f}"
        )

        df_abs = np.array(self.osys.mapper.delta_f_abs(phi_view), dtype=np.float32, copy=True)
        self.im_df.set_data(df_abs)
        df_max = float(df_abs.max())
        df_mean = float(df_abs.mean())
        p90 = float(np.percentile(df_abs, 90))
        target = max(120.0, min(1800.0, p90 * 1.8 + 80.0, df_max * 1.1 + 50.0))
        if target > self._df_clim:
            self._df_clim = 0.55 * self._df_clim + 0.45 * target
        else:
            self._df_clim = 0.90 * self._df_clim + 0.10 * target
        self.im_df.set_clim(0.0, max(120.0, self._df_clim))
        self.df_readout.set_text(
            f"|Δf| max {df_max:.0f} Hz   mean {df_mean:.0f}\n"
            f"scale 0–{self._df_clim:.0f} Hz   drive {self.osys.field._drive:.2f}"
        )

        hist = self.osys.csi.motion_history if self.osys.csi else []
        if hist:
            ys = np.asarray(hist[-200:], dtype=float)
            self.hist_line.set_data(np.arange(len(ys)), ys)
            self.ax_hist.set_xlim(0, max(50, len(ys)))
        for rect, h in zip(self.bars, self._sub_bars()):
            rect.set_height(float(h))

        self.status.set_text(
            f"frame={self._frame}  motion={e:.3f}  tracks={len(tracks)}  "
            f"|Δf|_max={df_max:.0f}Hz  drive={self.osys.field._drive:.2f}  "
            f"pkts={self.osys.csi_packets}  t={self.osys.t:.1f}s"
        )

        now = time.time()
        if now - self._last_log > 1.0:
            self._last_log = now
            print(
                f"[live] |Δf|_max={df_max:.1f}  drive={self.osys.field._drive:.2f}  "
                f"motion={e:.3f}  tracks={len(tracks)}  pkts={self.osys.csi_packets}"
            )

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend"); self.osys.close(); return
        print(f"[dashboard] backend={_BACKEND}  csi={'on' if self.osys.csi_enabled else 'off'}")
        plt.show(block=False)
        self.fig.canvas.draw()
        try:
            while self._running and plt.fignum_exists(self.fig.number):
                self.tick()
                plt.pause(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            try:
                self.osys.save(force=True)
            except Exception:
                pass
            self.osys.close()
            try:
                plt.close(self.fig)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description="Echo Grid live dashboard (CSI default on)")
    p.add_argument("--body", nargs="?", const="", default=None,
                   help="attach ultrasonic body serial (optional port)")
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=4210,
                   help="CSI UDP port (default 4210)")
    p.add_argument("--no-csi", action="store_true", help="disable CSI input")
    p.add_argument("--drive", action="store_true", help="drive body emitters from field")
    p.add_argument("--no-loop", action="store_true", help="disable closed-loop feedback")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()

    csi_port = None if a.no_csi else a.csi
    LiveDashboard(a.size, a.body, csi_port, a.drive, not a.no_loop).run()


if __name__ == "__main__":
    main()
