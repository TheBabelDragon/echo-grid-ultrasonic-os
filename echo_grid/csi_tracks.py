"""
CSI multi-target tracker — high-end classical ML stack.

Kalman (CV) + Hungarian assignment + CSI feature embedding.
Real-time, no heavyweight DNN dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian algorithm (minimize). Returns list of (row, col)."""
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    # pad to square
    C = np.full((n, n), 1e6, dtype=float)
    C[:n_rows, :n_cols] = cost

    # Step 1: row reduce
    C = C - C.min(axis=1, keepdims=True)
    # Step 2: col reduce
    C = C - C.min(axis=0, keepdims=True)

    star = np.zeros((n, n), dtype=bool)
    prime = np.zeros((n, n), dtype=bool)
    row_cov = np.zeros(n, dtype=bool)
    col_cov = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if C[i, j] == 0 and not row_cov[i] and not col_cov[j]:
                star[i, j] = True
                row_cov[i] = True
                col_cov[j] = True
    row_cov[:] = False
    col_cov[:] = False

    def cover_starred():
        col_cov[:] = star.any(axis=0)

    cover_starred()

    def find_uncovered_zero():
        for i in range(n):
            if row_cov[i]:
                continue
            for j in range(n):
                if not col_cov[j] and C[i, j] == 0:
                    return i, j
        return None

    max_iter = n * n * 4
    it = 0
    while col_cov.sum() < n and it < max_iter:
        it += 1
        z = find_uncovered_zero()
        if z is None:
            # augment by smallest uncovered value
            mask = ~row_cov[:, None] & ~col_cov[None, :]
            if not mask.any():
                break
            m = C[mask].min()
            C[row_cov, :] += m
            C[:, ~col_cov] -= m
            continue
        i, j = z
        prime[i, j] = True
        star_cols = np.where(star[i])[0]
        if len(star_cols) == 0:
            # augment path
            while True:
                star[i, j] = True
                # find star in column j
                star_rows = np.where(star[:, j])[0]
                star_rows = [r for r in star_rows if r != i and star[r, j]]
                # simpler path construction
                star_in_col = None
                for r in range(n):
                    if star[r, j] and r != i:
                        star_in_col = r
                        break
                if star_in_col is None:
                    break
                star[star_in_col, j] = False
                # find prime in that row
                prime_cols = np.where(prime[star_in_col])[0]
                if len(prime_cols) == 0:
                    break
                i = star_in_col
                j = int(prime_cols[0])
            prime[:, :] = False
            row_cov[:] = False
            col_cov[:] = False
            cover_starred()
        else:
            row_cov[i] = True
            col_cov[star_cols[0]] = False

    pairs = []
    for i in range(n_rows):
        for j in range(n_cols):
            if star[i, j]:
                pairs.append((i, j))
    return pairs


def _find_peaks(vals: np.ndarray, max_peaks: int = 4) -> List[Tuple[int, float]]:
    n = len(vals)
    if n < 5:
        return []
    peaks = []
    for i in range(1, n - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            left = vals[i]
            for j in range(i - 1, -1, -1):
                left = min(left, vals[j])
                if vals[j] > vals[i]:
                    break
            right = vals[i]
            for j in range(i + 1, n):
                right = min(right, vals[j])
                if vals[j] > vals[i]:
                    break
            prom = float(vals[i] - max(left, right))
            if prom > 0.03 and vals[i] > 0.10:
                peaks.append((i, prom * float(vals[i])))
    peaks.sort(key=lambda t: t[1], reverse=True)
    return peaks[:max_peaks]


def _csi_feature(vals: np.ndarray) -> np.ndarray:
    """Compact embedding for association."""
    if len(vals) == 0:
        return np.zeros(8, dtype=float)
    # resample to 8 bins
    idx = np.linspace(0, len(vals) - 1, 8)
    feat = np.interp(idx, np.arange(len(vals)), vals)
    norm = np.linalg.norm(feat) + 1e-6
    return feat / norm


@dataclass
class Detection:
    x: float
    y: float
    energy: float
    feature: np.ndarray
    source: str


class KalmanTrack:
    """2D constant-velocity Kalman: state [x, y, vx, vy]."""

    _next_id = 1

    def __init__(self, x: float, y: float, energy: float, feature: np.ndarray, source: str):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.track_id = f"T{self.id}"
        self.source = source
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 0.15
        self.energy = energy
        self.feature = feature.copy()
        self.confidence = 0.35
        self.hits = 1
        self.misses = 0
        self.last_seen = time.time()
        self.state = "idle"  # idle | move | surge
        self.rssi = -90.0

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    def predict(self, dt: float):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        q = 0.08 + 0.25 * (1.0 - self.confidence)
        Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=float) * q
        self.x = F @ self.x
        self.x[0] = min(1.0, max(0.0, self.x[0]))
        self.x[1] = min(1.0, max(0.0, self.x[1]))
        self.P = F @ self.P @ F.T + Q
        self.misses += 1

    def update(self, det: Detection):
        z = np.array([det.x, det.y], dtype=float)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        r = 0.04 + 0.12 * (1.0 - min(1.0, det.energy))
        R = np.eye(2) * r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T * 0.0
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.x[0] = min(1.0, max(0.0, self.x[0]))
        self.x[1] = min(1.0, max(0.0, self.x[1]))

        self.energy = 0.65 * self.energy + 0.35 * det.energy
        self.feature = 0.70 * self.feature + 0.30 * det.feature
        fn = np.linalg.norm(self.feature) + 1e-6
        self.feature /= fn

        self.hits += 1
        self.misses = 0
        self.last_seen = time.time()

        # confidence: evidence vs uncertainty
        pos_unc = float(self.P[0, 0] + self.P[1, 1])
        self.confidence = float(min(0.99, max(0.15,
            0.2 + 0.15 * min(self.hits, 10) / 10
            + 0.45 * self.energy
            + 0.2 * (1.0 / (1.0 + 5.0 * pos_unc))
        )))

        sp = self.speed
        if self.energy > 0.72 and sp > 0.15:
            self.state = "surge"
        elif sp > 0.08 or self.energy > 0.45:
            self.state = "move"
        else:
            self.state = "idle"


class TrackStore:
    def __init__(self, max_tracks: int = 8, ttl_s: float = 2.8):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: List[KalmanTrack] = []
        self._last = time.time()

    def _packet_energy(self, pkt: Dict[str, Any]) -> float:
        for key in ("movement_intensity", "activity"):
            if key in pkt and pkt[key] is not None:
                try:
                    return float(min(1.0, max(0.0, float(pkt[key]))))
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _detections_from_packet(self, pkt: Dict[str, Any]) -> List[Detection]:
        node = str(pkt.get("node") or "csi")
        energy = self._packet_energy(pkt)
        raw = pkt.get("csi") or []
        try:
            vals = np.array([float(v) for v in raw], dtype=float)
        except (TypeError, ValueError):
            vals = np.zeros(0)

        if len(vals) == 0:
            vals = np.array([energy] * 16)

        feat = _csi_feature(vals)
        dets: List[Detection] = []

        # centroid detection from spectral mass
        if len(vals) >= 4:
            w = vals / (vals.sum() + 1e-6)
            idx = np.arange(len(vals))
            cx = float(0.15 + 0.70 * np.dot(w, idx) / max(1, len(vals) - 1))
            cy = float(0.25 + 0.50 * energy)
        else:
            cx, cy = 0.5, 0.45
        dets.append(Detection(cx, cy, max(energy, 0.05), feat, node))

        # peak detections
        if energy > 0.12 and len(vals) >= 12:
            for rank, (i, score) in enumerate(_find_peaks(vals, max_peaks=4)):
                e = float(min(1.0, vals[i]))
                if e < 0.12:
                    continue
                px = float(0.12 + 0.76 * (i / max(1, len(vals) - 1)))
                py = float(0.22 + 0.55 * e)
                # local feature slice
                lo, hi = max(0, i - 2), min(len(vals), i + 3)
                local = _csi_feature(vals[lo:hi])
                dets.append(Detection(px, py, e, local, f"{node}:p{rank}"))

        return dets

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        now = time.time()
        dt = max(1e-3, min(0.25, now - self._last))
        self._last = now

        for tr in self.tracks:
            tr.predict(dt)

        dets = self._detections_from_packet(pkt)
        try:
            rssi = float(pkt.get("rssi", -90))
        except (TypeError, ValueError):
            rssi = -90.0

        if not self.tracks and not dets:
            return

        if self.tracks and dets:
            cost = np.zeros((len(self.tracks), len(dets)), dtype=float)
            for i, tr in enumerate(self.tracks):
                tx, ty = tr.pos
                for j, d in enumerate(dets):
                    pos_d = math.hypot(tx - d.x, ty - d.y)
                    feat_d = float(np.linalg.norm(tr.feature - d.feature))
                    # gate impossible associations
                    if pos_d > 0.40:
                        cost[i, j] = 1e5
                    else:
                        cost[i, j] = 1.0 * pos_d + 0.55 * feat_d + 0.15 * abs(tr.energy - d.energy)

            pairs = _hungarian(cost)
            assigned_tr = set()
            assigned_det = set()
            for i, j in pairs:
                if i >= len(self.tracks) or j >= len(dets):
                    continue
                if cost[i, j] > 0.55:
                    continue
                self.tracks[i].update(dets[j])
                self.tracks[i].rssi = rssi
                assigned_tr.add(i)
                assigned_det.add(j)

            # new tracks
            for j, d in enumerate(dets):
                if j in assigned_det:
                    continue
                if d.energy < 0.10:
                    continue
                if len(self.tracks) >= self.max_tracks:
                    self._drop_weakest()
                if len(self.tracks) >= self.max_tracks:
                    break
                tr = KalmanTrack(d.x, d.y, d.energy, d.feature, d.source)
                tr.rssi = rssi
                self.tracks.append(tr)
        elif dets:
            for d in dets:
                if d.energy < 0.10:
                    continue
                if len(self.tracks) >= self.max_tracks:
                    break
                tr = KalmanTrack(d.x, d.y, d.energy, d.feature, d.source)
                tr.rssi = rssi
                self.tracks.append(tr)

        self._expire()

    def _drop_weakest(self):
        if not self.tracks:
            return
        weak = min(self.tracks, key=lambda t: (t.confidence, t.hits, t.energy))
        self.tracks.remove(weak)

    def _expire(self):
        now = time.time()
        alive = []
        for t in self.tracks:
            # coast a few misses; then drop
            if now - t.last_seen > self.ttl_s:
                continue
            if t.misses > 12 and t.confidence < 0.4:
                continue
            alive.append(t)
        self.tracks = alive

    def active(self) -> List[KalmanTrack]:
        self._expire()
        out = [t for t in self.tracks if t.hits >= 1 and t.confidence > 0.22]
        return sorted(out, key=lambda t: t.confidence * t.energy, reverse=True)
