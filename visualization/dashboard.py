#!/usr/bin/env python3
"""Echo Grid live — intelligent visualization of fused radio belief + φ."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
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
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D

from echo_grid.core import EchoGridOS

TRACK_COLORS = ["#ff4d6d", "#4cc9f0", "#f4a261", "#a0e8af", "#c77dff", "#ffe066"]


def _peaks(grid: np.ndarray, max_peaks: int = 5, floor: float = 0.15):
    """Local maxima in unit grid → list of (x, y, value) in [0,1]²."""
    if grid is None or grid.size == 0:
        return []
    g = grid.astype(float)
    peak = float(g.max())
    if peak < 1e-6:
        return []
    h, w = g.shape
    work = g.copy()
    out = []
    for _ in range(max_peaks):
        j, i = np.unravel_index(int(np.argmax(work)), work.shape)
        v = float(work[j, i])
        if v < floor * peak:
            break
        out.append(((i + 0.5) / w, (j + 0.5) / h, v / peak))
        j0, j1 = max(0, j - 2), min(h, j + 3)
        i0, i1 = max(0, i - 2), min(w, i + 3)
        work[j0:j1, i0:i1] = 0.0
    return out


class LiveDashboard:
    def __init__(self, size=16, body_port=None, csi_port=4210, drive=False, closed_loop=True):
        self.osys = EchoGridOS(size=size, body_port=body_port, csi_port=csi_port)
        self.drive = drive
        self.closed_loop = closed_loop
        self.size = size
        self._frame = 0
        self._last_log = 0.0
        self._running = True
        self._df_clim = 350.0
        self._conf_hist: deque = deque(maxlen=120)
        self._agree_hist: deque = deque(maxlen=120)
        self._mode_marks: deque = deque(maxlen=8)  # (frame, label)
        self._last_agree = None

        self.fig = plt.figure(figsize=(13, 9.5))
        gs = self.fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28,
                                   left=0.06, right=0.98, top=0.92, bottom=0.06)
        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_csi_sp = self.fig.add_subplot(gs[0, 1])
        self.ax_df = self.fig.add_subplot(gs[1, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        z = np.zeros((size, size), dtype=np.float32)

        self.im_phi = self.ax_phi.imshow(
            z.copy(), cmap="viridis", vmin=-1, vmax=1, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_phi.set_title("1 · φ + tracks + belief peaks", fontsize=11)
        self.fig.colorbar(self.im_phi, ax=self.ax_phi, fraction=0.046, pad=0.04)

        # Belief peak markers on φ
        self.peak_marks = []
        for _ in range(5):
            circ = Circle((0, 0), 0.04, fill=False, edgecolor="#ffffff",
                          lw=1.8, ls="--", alpha=0, zorder=5)
            self.ax_phi.add_patch(circ)
            self.peak_marks.append(circ)

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
        self.ax_csi_sp.set_title("2 · Fused belief (dynamic)", fontsize=11)
        self.fig.colorbar(self.im_csi, ax=self.ax_csi_sp, fraction=0.046, pad=0.04)
        self.csi_readout = self.ax_csi_sp.text(
            0.02, 0.98, "waiting…", transform=self.ax_csi_sp.transAxes,
            ha="left", va="top", fontsize=8, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.8),
        )
        # room memory inset
        self.ax_room = self.ax_csi_sp.inset_axes([0.62, 0.02, 0.36, 0.36])
        self.im_room = self.ax_room.imshow(
            z.copy(), cmap="cividis", vmin=0, vmax=1, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_room.set_title("room", fontsize=7, color="#ccc", pad=1)
        self.ax_room.set_xticks([])
        self.ax_room.set_yticks([])

        self.im_df = self.ax_df.imshow(
            z.copy(), cmap="inferno", vmin=0, vmax=400, origin="lower", extent=[0, 1, 0, 1],
        )
        self.ax_df.set_title("3 · Actuator |Δf| (belief-held)", fontsize=11)
        self.cbar_df = self.fig.colorbar(self.im_df, ax=self.ax_df, fraction=0.046, pad=0.04)
        self.df_readout = self.ax_df.text(
            0.02, 0.98, "|Δf| max —", transform=self.ax_df.transAxes,
            ha="left", va="top", fontsize=8, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#222", alpha=0.8),
        )

        self.ax_hist.set_title("4 · Motion + fuse confidence", fontsize=11)
        self.ax_hist.set_xlim(0, 120)
        self.ax_hist.set_ylim(0, 1.05)
        self.hist_line, = self.ax_hist.plot([], [], color="#00d4aa", lw=2.0, label="motion")
        self.conf_line, = self.ax_hist.plot([], [], color="#c77dff", lw=1.6, alpha=0.9, label="fuse conf")
        self.agree_fill = self.ax_hist.fill_between([0], [0], [0], color="#2a9d8a", alpha=0.15)
        self.ax_hist.legend(loc="upper left", fontsize=7, framealpha=0.6)
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
            bits.append("csi+fuse")
        self.fig.suptitle(f"Echo Grid  ·  {'+'.join(bits) or 'idle'}  ·  intelligent HUD", fontsize=13)
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=8, family="monospace")
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "_running", False))
        # click inject on φ
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes is not self.ax_phi or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.osys.touch(x, y, strength=1.0)
            print(f"[ui] touch inject ({x:.2f},{y:.2f})")

    def _sub_bars(self):
        if not self.osys.csi or not self.osys.csi.last_packet:
            return np.zeros(16)
        try:
            vals = np.array([float(x) for x in (self.osys.csi.last_packet.get("csi") or [])], dtype=float)
        except (TypeError, ValueError):
            return np.zeros(16)
        if len(vals) == 0:
            return np.zeros(16)
        idx = np.linspace(0, len(vals) - 1, 16)
        return np.clip(np.interp(idx, np.arange(len(vals)), vals), 0, 1)

    def _node_legend(self) -> str:
        if not self.osys.csi:
            return ""
        nodes = getattr(self.osys.csi, "nodes", {}) or {}
        if not nodes:
            return "nodes: —"
        parts = []
        for nid, info in list(nodes.items())[:4]:
            band = info.get("band", "?")
            short = nid[-12:] if len(nid) > 12 else nid
            parts.append(f"{short}:{band}")
        return " | ".join(parts)

    def tick(self):
        if not self._running:
            return
        self._frame += 1
        phi = self.osys.step(drive_body=self.drive, closed_loop=self.closed_loop)
        phi_view = np.array(phi, dtype=np.float32, copy=True)

        # --- φ ---
        self.im_phi.set_data(phi_view)
        lo, hi = np.percentile(phi_view, [5, 95])
        span = max(0.12, float(hi - lo) * 0.55 + 1e-6)
        mid = float(phi_view.mean())
        self.im_phi.set_clim(mid - 2.5 * span, mid + 2.5 * span)

        # belief peaks overlay
        occ = None
        room = None
        if self.osys.csi and getattr(self.osys.csi, "fuser", None):
            fused = self.osys.csi.fuser.last_fused
            if fused is not None:
                occ = fused.occupancy
                room = fused.room
        peaks = _peaks(occ, max_peaks=5) if occ is not None else []
        for i, mark in enumerate(self.peak_marks):
            if i < len(peaks):
                x, y, v = peaks[i]
                mark.center = (x, y)
                mark.set_radius(0.03 + 0.05 * v)
                mark.set_alpha(0.35 + 0.55 * v)
                mark.set_edgecolor("#ffffff" if self.osys.fuse_agreed else "#ffaa00")
            else:
                mark.set_alpha(0)

        tracks = self.osys.csi.active_tracks() if self.osys.csi else []
        fuse_scale = 0.7 + 0.5 * self.osys.fuse_conf if self.osys.fuse_agreed else 0.55
        for i in range(len(TRACK_COLORS)):
            ring, dot, lab = self.track_rings[i], self.track_dots[i], self.track_labels[i]
            if i < len(tracks):
                tr = tracks[i]
                x, y = tr.pos
                a = min(1.0, 0.3 + tr.confidence * 0.7) * fuse_scale
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

        # --- fused belief ---
        spatial = (
            np.array(self.osys.csi.spatial_map(self.size), dtype=np.float32, copy=True)
            if self.osys.csi else np.zeros((self.size, self.size), dtype=np.float32)
        )
        self.im_csi.set_data(spatial)
        self.im_csi.set_clim(0.0, max(0.25, float(spatial.max()) + 1e-9))

        if room is not None:
            r = np.array(room, dtype=np.float32)
            self.im_room.set_data(r)
            self.im_room.set_clim(0.0, max(0.2, float(r.max()) + 1e-9))

        e = float(self.osys.last_csi_energy)
        agreed = self.osys.fuse_agreed
        self._conf_hist.append(self.osys.fuse_conf)
        self._agree_hist.append(1.0 if agreed else 0.0)
        if self._last_agree is not None and agreed != self._last_agree:
            self._mode_marks.append((self._frame, "AGR" if agreed else "sgl"))
        self._last_agree = agreed

        self.csi_readout.set_text(
            f"motion {e:.2f}  pkts {self.osys.csi_packets}\n"
            f"src {self.osys.fuse_sources}  bands {self.osys.fuse_bands}  "
            f"agree {'Y' if agreed else 'n'}  conf {self.osys.fuse_conf:.2f}\n"
            f"{self._node_legend()}"
        )

        # --- |Δf| ---
        df_abs = np.array(self.osys.actuator_map(phi_view), dtype=np.float32, copy=True)
        self.im_df.set_data(df_abs)
        df_max = float(df_abs.max())
        df_mean = float(df_abs.mean())
        p90 = float(np.percentile(df_abs, 90)) if df_max > 0 else 0.0
        target = max(150.0, min(2000.0, max(p90 * 1.7, df_max * 1.15) + 60.0))
        if target > self._df_clim:
            self._df_clim = 0.5 * self._df_clim + 0.5 * target
        else:
            self._df_clim = 0.93 * self._df_clim + 0.07 * target
        self.im_df.set_clim(0.0, max(150.0, self._df_clim))
        self.df_readout.set_text(
            f"|Δf| max {df_max:.0f} Hz   mean {df_mean:.0f}\n"
            f"peaks {len(peaks)}  agree {'Y' if agreed else 'n'}  "
            f"drive {self.osys.field._drive:.2f}"
        )

        # --- history ---
        hist = self.osys.csi.motion_history if self.osys.csi else []
        if hist:
            ys = np.asarray(hist[-120:], dtype=float)
            xs = np.arange(len(ys))
            self.hist_line.set_data(xs, ys)
            self.ax_hist.set_xlim(0, max(40, len(ys)))
        if self._conf_hist:
            cs = np.asarray(self._conf_hist, dtype=float)
            self.conf_line.set_data(np.arange(len(cs)), cs)
        # agree shading via vertical markers
        for art in list(self.ax_hist.lines):
            if getattr(art, "_echo_mark", False):
                art.remove()
        for fr, lab in self._mode_marks:
            # map frame to x roughly if hist length known
            x = max(0, len(hist[-120:]) - (self._frame - fr))
            if 0 <= x <= 120:
                ln = self.ax_hist.axvline(x, color="#ffe066" if lab == "AGR" else "#888",
                                          lw=0.9, alpha=0.7)
                ln._echo_mark = True

        for rect, h in zip(self.bars, self._sub_bars()):
            rect.set_height(float(h))
            # tint bars by agreement
            rect.set_color("#4cc9f0" if not agreed else "#c77dff")

        self.status.set_text(
            f"frame={self._frame}  motion={e:.3f}  tracks={len(tracks)}  peaks={len(peaks)}  "
            f"fuse={self.osys.fuse_sources}s/{self.osys.fuse_bands}b agree={'Y' if agreed else 'n'}  "
            f"|Δf|_max={df_max:.0f}Hz  click-φ to inject"
        )

        now = time.time()
        if now - self._last_log > 1.0:
            self._last_log = now
            print(
                f"[live] |Δf|_max={df_max:.1f}  motion={e:.3f}  peaks={len(peaks)}  "
                f"fuse={self.osys.fuse_sources}s/{self.osys.fuse_bands}b agree={'Y' if agreed else 'n'}  "
                f"tracks={len(tracks)}"
            )

    def run(self):
        if _BACKEND == "Agg":
            print("No GUI backend"); self.osys.close(); return
        print(f"[dashboard] backend={_BACKEND}  click panel-1 to inject")
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
    p = argparse.ArgumentParser()
    p.add_argument("--body", nargs="?", const="", default=None)
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=4210)
    p.add_argument("--no-csi", action="store_true")
    p.add_argument("--drive", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()
    csi_port = None if a.no_csi else a.csi
    LiveDashboard(a.size, a.body, csi_port, a.drive, not a.no_loop).run()


if __name__ == "__main__":
    main()
