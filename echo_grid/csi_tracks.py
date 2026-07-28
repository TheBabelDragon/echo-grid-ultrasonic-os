"""
Real CSI multi-target tracker.

Background model → residual peaks → gated Kalman tracks.
No synthetic detections — only structure present in live CSI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _smooth(vals: np.ndarray, k: int = 3) -> np.ndarray:
    if len(vals) < k:
        return vals.copy()
    pad = k // 2
    xp = np.pad(vals, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode="valid")


def _peaks(residual: np.ndarray, max_peaks: int = 5) -> List[Tuple[int, float]]:
    """Peaks on residual (above background)."""
    n = len(residual)
    if n < 5:
        return []
    s = _smooth(residual, 3)
    out = []
    for i in range(1, n - 1):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.06:
            # prominence vs neighbors
            prom = s[i] - 0.5 * (s[i - 1] + s[i + 1])
            if prom > 0.02:
                out.append((i, float(s[i])))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:max_peaks]


@dataclass
class Detection:
    x: float
    y: float
    energy: float
    feature: np.ndarray


class KalmanTrack:
    _id = 1

    def __init__(self, x: float, y: float, energy: float, feature: np.ndarray):
        self.track_id = f"T{KalmanTrack._id}"
        KalmanTrack._id += 1
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 0.2
        self.energy = float(energy)
        self.feature = feature.astype(float).copy()
        n = np.linalg.norm(self.feature) + 1e-9
        self.feature /= n
        self.confidence = 0.4
        self.hits = 1
        self.misses = 0
        self.last_seen = time.time()
        self.state = "idle"
        self.rssi = -90.0

    @property
    def pos(self) -> Tuple[float, float]:
        return float(np.clip(self.x[0], 0, 1)), float(np.clip(self.x[1], 0, 1))

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
        q = 0.05 + 0.2 * (1.0 - self.confidence)
        Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=float)
        self.x = F @ self.x
        self.x[0] = float(np.clip(self.x[0], 0, 1))
        self.x[1] = float(np.clip(self.x[1], 0, 1))
        self.P = F @ self.P @ F.T + Q
        self.misses += 1
        # mild confidence decay while coasting
        self.confidence *= 0.985

    def update(self, det: Detection):
        z = np.array([det.x, det.y], dtype=float)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * (0.03 + 0.1 * (1.0 - min(1.0, det.energy)))
        innov = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P
        self.x[0] = float(np.clip(self.x[0], 0, 1))
        self.x[1] = float(np.clip(self.x[1], 0, 1))

        self.energy = 0.6 * self.energy + 0.4 * det.energy
        self.feature = 0.75 * self.feature + 0.25 * det.feature
        self.feature /= (np.linalg.norm(self.feature) + 1e-9)

        self.hits += 1
        self.misses = 0
        self.last_seen = time.time()

        unc = float(self.P[0, 0] + self.P[1, 1])
        self.confidence = float(np.clip(
            0.15 + 0.5 * self.energy + 0.2 * min(self.hits, 12) / 12 + 0.15 / (1 + 8 * unc),
            0.05, 0.99,
        ))

        sp = self.speed
        if self.energy > 0.65 and sp > 0.12:
            self.state = "surge"
        elif sp > 0.06 or self.energy > 0.35:
            self.state = "move"
        else:
            self.state = "idle"


class TrackStore:
    def __init__(self, max_tracks: int = 6, ttl_s: float = 2.2):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: List[KalmanTrack] = []
        self._bg: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self._last = time.time()
        self.motion_energy = 0.0

    def _vals(self, pkt: Dict[str, Any]) -> np.ndarray:
        raw = pkt.get("csi") or []
        try:
            v = np.array([float(x) for x in raw], dtype=float)
        except (TypeError, ValueError):
            v = np.zeros(0)
        if len(v) == 0:
            # fall back to scalar activity as flat profile
            e = 0.0
            for k in ("movement_intensity", "activity"):
                if k in pkt:
                    try:
                        e = float(pkt[k])
                    except (TypeError, ValueError):
                        pass
            v = np.full(32, e, dtype=float)
        return v

    def _packet_motion(self, pkt: Dict[str, Any], vals: np.ndarray) -> float:
        for k in ("movement_intensity", "activity"):
            if k in pkt and pkt[k] is not None:
                try:
                    return float(np.clip(float(pkt[k]), 0, 1))
                except (TypeError, ValueError):
                    pass
        if self._prev is not None and len(self._prev) == len(vals):
            return float(np.clip(np.mean(np.abs(vals - self._prev)) * 5.0, 0, 1))
        return float(np.clip(np.std(vals) * 2.0, 0, 1))

    def _detections(self, vals: np.ndarray, motion: float) -> List[Detection]:
        # update background slowly
        if self._bg is None or len(self._bg) != len(vals):
            self._bg = vals.copy()
        else:
            self._bg = 0.97 * self._bg + 0.03 * vals

        residual = np.clip(vals - self._bg, 0, None)
        # temporal residual
        if self._prev is not None and len(self._prev) == len(vals):
            residual = residual + 0.65 * np.clip(np.abs(vals - self._prev), 0, None)

        residual = _smooth(residual, 3)
        rmax = float(residual.max()) + 1e-9
        residual = residual / rmax

        dets: List[Detection] = []
        peaks = _peaks(residual, max_peaks=5)

        # If motion is present but peaks weak, still emit spectral centroid
        if motion > 0.08:
            w = residual / (residual.sum() + 1e-9)
            idx = np.arange(len(residual))
            cx = float(0.12 + 0.76 * np.dot(w, idx) / max(1, len(residual) - 1))
            cy = float(0.20 + 0.55 * motion)
            feat = residual[:: max(1, len(residual) // 8)][:8].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(cx, cy, max(motion, float(residual.mean())), feat))

        for i, strength in peaks:
            if strength < 0.12 and motion < 0.1:
                continue
            e = float(np.clip(0.35 * strength + 0.65 * motion, 0, 1))
            if e < 0.08:
                continue
            x = float(0.10 + 0.80 * (i / max(1, len(residual) - 1)))
            y = float(0.18 + 0.60 * e)
            lo, hi = max(0, i - 3), min(len(residual), i + 4)
            feat = residual[lo:hi].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            else:
                feat = feat[:8]
            feat = feat / (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(x, y, e, feat))

        # de-duplicate close detections
        dets = self._nms(dets, min_dist=0.12)
        return dets

    def _nms(self, dets: List[Detection], min_dist: float) -> List[Detection]:
        dets = sorted(dets, key=lambda d: d.energy, reverse=True)
        kept: List[Detection] = []
        for d in dets:
            if all(math.hypot(d.x - k.x, d.y - k.y) >= min_dist for k in kept):
                kept.append(d)
        return kept

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        now = time.time()
        dt = float(np.clip(now - self._last, 1e-3, 0.3))
        self._last = now

        vals = self._vals(pkt)
        motion = self._packet_motion(pkt, vals)
        self.motion_energy = motion

        for tr in self.tracks:
            tr.predict(dt)

        dets = self._detections(vals, motion)
        self._prev = vals.copy()

        try:
            rssi = float(pkt.get("rssi", -90))
        except (TypeError, ValueError):
            rssi = -90.0

        if self.tracks and dets:
            # gated greedy (stable; Hungarian optional later)
            used_d = set()
            used_t = set()
            pairs = []
            for ti, tr in enumerate(self.tracks):
                tx, ty = tr.pos
                for di, d in enumerate(dets):
                    pd = math.hypot(tx - d.x, ty - d.y)
                    fd = float(np.linalg.norm(tr.feature - d.feature))
                    if pd > 0.32:
                        continue
                    cost = 1.1 * pd + 0.45 * fd
                    pairs.append((cost, ti, di))
            pairs.sort()
            for cost, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                if cost > 0.5:
                    continue
                self.tracks[ti].update(dets[di])
                self.tracks[ti].rssi = rssi
                used_t.add(ti)
                used_d.add(di)

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                if d.energy < 0.12:
                    continue
                if len(self.tracks) >= self.max_tracks:
                    self._drop_weakest()
                if len(self.tracks) >= self.max_tracks:
                    break
                tr = KalmanTrack(d.x, d.y, d.energy, d.feature)
                tr.rssi = rssi
                self.tracks.append(tr)
        elif dets:
            for d in dets:
                if d.energy < 0.12:
                    continue
                if len(self.tracks) >= self.max_tracks:
                    break
                tr = KalmanTrack(d.x, d.y, d.energy, d.feature)
                tr.rssi = rssi
                self.tracks.append(tr)

        self._expire()

    def _drop_weakest(self):
        if not self.tracks:
            return
        w = min(self.tracks, key=lambda t: (t.confidence, t.hits, t.energy))
        self.tracks.remove(w)

    def _expire(self):
        now = time.time()
        self.tracks = [
            t for t in self.tracks
            if (now - t.last_seen) <= self.ttl_s and not (t.misses > 15 and t.confidence < 0.35)
        ]

    def active(self) -> List[KalmanTrack]:
        self._expire()
        return sorted(
            [t for t in self.tracks if t.confidence > 0.25 and t.hits >= 2],
            key=lambda t: t.confidence * t.energy,
            reverse=True,
        )
