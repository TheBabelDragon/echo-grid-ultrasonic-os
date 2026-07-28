#!/usr/bin/env python3
"""
Echo Grid — live 2x2 stack (Tk-safe animation)

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
        self._frame = 0
        self._last_df_max = -1.0
        self._last_log = 0.0
        self._running = True
        self._ani: FuncAnimation | None = None

        self.fig = plt.figure(figsize=(12, 9))
        gs = self.fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_csi_sp = self.fig.add_subplot(gs[0, 1])
        self.ax_df = self.fig.add_subplot(gs[1, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        z = np.zeros((size, size), dtype=np.float32)

        self.im_phi = self.ax_phi.imshow(
            z.copy(), cmap="viridis", vmin=-1, vmax=1, origin="lower",
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

        self.im_csi = self.ax_csi_sp.imshow(
            z.copy(), cmap="magma", vmin=0, vmax=1, origin="lower",
            extent=[0, 1, 0, 1], animated=True,
        )
        self.ax_csi_sp.set_title("2 · CSI spatial map (RF residual)", fontsize=11)
        self.fig.colorbar(self.im_csi, ax=self.ax_csi_sp, fraction=0.046, pad=0.04)
        self.csi_readout = self.ax_csi_sp.text(
            0.02, 0.98, "waiting…", transform=self.ax_csi_sp.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        self.im_df = self.ax_df.imshow(
            z.copy(), cmap="coolwarm", vmin=-500, vmax=500, origin="lower",
            extent=[0, 1, 0, 1], animated=True,
        )
        self.ax_df.set_title("3 · Actuator Δf  (dynamic)", fontsize=11)
        self.cbar_df = self.fig.colorbar(self.im_df, ax=self.ax_df, fraction=0.046, pad=0.04)
        self.cbar_df.set_label("Hz from 40 kHz")
        self.df_readout = self.ax_df.text(
            0.02, 0.98, "Δf max —", transform=self.ax_df.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.75),
        )

        self.ax_hist.set_title("4 · Motion history + subcarriers", fontsize=11)
        self.ax_hist.set_xlim(0, 200)
        self.ax_hist.set_ylim(0, 1.05)
        self.ax_hist.set_xlabel("recent packets")
        self.ax_hist.set_ylabel("motion")
        self.hist_line, = self.ax_hist.plot([], [], color="#00d4aa", lw=2.0)
        self.ax_hist.axhline(0.3, color="#555", ls="--", lw=0.8)
        self.ax_hist.axhline(0.7, color="#555", ls=":", lw=0.8)

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
        self.fig.suptitle(f"Echo Grid  ·  live  ·  {'+'.join(bits)}", fontsize=13)
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9, family="monospace")

        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event=None):
        self._running = False
        ani = self._ani
        if ani is not None:
            try:
                if ani.event_source is not None:
                    ani.event_source.stop()
            except Exception:
                pass
            try:
                ani._stop()  # type: ignore[attr-defined]
            except Exception:
                pass
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
        if not self._running or self._ani is None:
            return []

        try:
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
                    ring.set_alpha(0)
                    dot.set_alpha(0)
                    lab.set_alpha(0)

            if self.osys.csi is not None:
                spatial = np.array(self.osys.csi.spatial_map(self.size), dtype=np.float32, copy=True)
            else:
                spatial = np.zeros((self.size, self.size), dtype=np.float32)
            self.im_csi.set_data(spatial)
            smax = float(spatial.max()) + 1e-9
            self.im_csi.set_clim(0.0, max(0.25, smax))
            e = float(self.osys.last_csi_energy)
            self.csi_readout.set_text(
                f"motion {e:.2f}   pkts {self.osys.csi_packets}\n"
                f"tracks {len(tracks)}   rssi {self.osys.csi.last_rssi if self.osys.csi else 0:.0f}"
            )

            df = np.array(self.osys.mapper.delta_f(phi_view), dtype=np.float32, copy=True)
            self.im_df.set_data(df)
            abs_df = np.abs(df)
            df_max = float(abs_df.max())
            df_mean = float(abs_df.mean())
            p95 = float(np.percentile(abs_df, 95))
            fclim = max(50.0, min(2500.0, max(p95 * 2.2, df_max * 1.15) + 25.0))
            self.im_df.set_clim(-fclim, fclim)
            self.df_readout.set_text(
                f"Δf max {df_max:.0f} Hz   mean {df_mean:.0f}\n"
                f"scale ±{fclim:.0f} Hz   frame {self._frame}"
            )

            hist = self.osys.csi.motion_history if self.osys.csi else []
            if hist:
                ys = np.asarray(hist[-200:], dtype=float)
                xs = np.arange(len(ys))
                self.hist_line.set_data(xs, ys)
                self.ax_hist.set_xlim(0, max(50, len(ys)))
            for rect, h in zip(self.bars, self._sub_bars()):
                rect.set_height(float(h))

            self.status.set_text(
                f"frame={self._frame}  motion={e:.3f}  tracks={len(tracks)}  "
                f"Δf_max={df_max:.0f}Hz  Δf_mean={df_mean:.0f}Hz  "
                f"pkts={self.osys.csi_packets}  t={self.osys.t:.1f}s"
            )

            now = time.time()
            if now - self._last_log > 1.0:
                self._last_log = now
                if abs(df_max - self._last_df_max) > 1.0 or self._frame < 5:
                    print(
                        f"[live] frame={self._frame}  Δf_max={df_max:.1f}Hz  "
                        f"Δf_mean={df_mean:.1f}Hz  motion={e:.3f}  pkts={self.osys.csi_packets}"
                    )
                self._last_df_max = df_max

        except Exception as ex:
            # never let a render error kill the Tk timer into NoneType interval
            if self._frame % 50 == 1:
                print(f"[dashboard] update error: {ex}")

        return []

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend")
            self.osys.close()
            return

        print(f"[dashboard] backend={_BACKEND}")

        # Strong references + repeat=True keeps event_source alive correctly
        self._ani = FuncAnimation(
            self.fig,
            self.update,
            interval=50,
            blit=False,
            cache_frame_data=False,
            repeat=True,
        )
        # Prevent GC of animation (root cause of event_source -> None)
        self.fig._echo_grid_ani = self._ani  # type: ignore[attr-defined]

        try:
            plt.show(block=True)
        finally:
            self._on_close()
            try:
                self.osys.save(force=True)
            except Exception:
                pass
            self.osys.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--body", nargs="?", const="", default=None)
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=None)
    p.add_argument("--drive", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()
    LiveDashboard(a.size, a.body, a.csi, a.drive, not a.no_loop).run()


if __name__ == "__main__":
    main()
