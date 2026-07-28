"""Smarter real CSI body detection + Kalman tracks."""

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
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _peaks(residual: np.ndarray, max_peaks: int = 4) -> List[Tuple[int, float]]:
    n = len(residual)
    if n < 5:
        return []
    s = _smooth(residual, 3)
    out = []
    for i in range(2, n - 2):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.08:
            left = min(s[i - 1], s[i - 2])
            right = min(s[i + 1], s[i + 2])
            prom = s[i] - 0.5 * (left + right)
            if prom > 0.025:
                out.append((i, float(s[i] * (1.0 + prom))))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:max_peaks]


@dataclass
class Detection:
    x: float
    y: float
    energy: float
    feature: np.ndarray
    prominence: float = 0.0


class KalmanTrack:
    _id = 1

    def __init__(self, x: float, y: float, energy: float, feature: np.ndarray):
        self.track_id = f"T{KalmanTrack._id}"
        KalmanTrack._id += 1
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 0.18
        self.energy = float(energy)
        self.feature = feature.astype(float).copy()
        self.feature /= (np.linalg.norm(self.feature) + 1e-9)
        self.confidence = 0.35
        self.hits = 1
        self.misses = 0
        self.last_seen = time.time()
        self.state = "idle"
        self.rssi = -90.0
        self.age = 0

    @property
    def pos(self) -> Tuple[float, float]:
        return float(np.clip(self.x[0], 0, 1)), float(np.clip(self.x[1], 0, 1))

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    def predict(self, dt: float):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = 0.04 + 0.18 * (1.0 - self.confidence)
        Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2],
        ], dtype=float)
        self.x = F @ self.x
        self.x[0] = float(np.clip(self.x[0], 0, 1))
        self.x[1] = float(np.clip(self.x[1], 0, 1))
        self.P = F @ self.P @ F.T + Q
        self.misses += 1
        self.age += 1
        self.confidence *= 0.98
        self.energy *= 0.99

    def update(self, det: Detection):
        z = np.array([det.x, det.y], dtype=float)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * (0.025 + 0.08 * (1.0 - min(1.0, det.energy)))
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

        self.energy = 0.55 * self.energy + 0.45 * det.energy
        self.feature = 0.7 * self.feature + 0.3 * det.feature
        self.feature /= (np.linalg.norm(self.feature) + 1e-9)

        self.hits += 1
        self.misses = 0
        self.last_seen = time.time()
        self.age += 1

        unc = float(self.P[0, 0] + self.P[1, 1])
        stab = 1.0 / (1.0 + self.speed)
        self.confidence = float(np.clip(
            0.12
            + 0.40 * self.energy
            + 0.22 * min(self.hits, 15) / 15
            + 0.15 * stab
            + 0.11 / (1 + 6 * unc),
            0.05, 0.99,
        ))

        sp = self.speed
        if self.energy > 0.6 and sp > 0.10:
            self.state = "surge"
        elif sp > 0.05 or self.energy > 0.32:
            self.state = "move"
        else:
            self.state = "idle"


class TrackStore:
    def __init__(self, max_tracks: int = 5, ttl_s: float = 2.0):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: List[KalmanTrack] = []
        self._bg: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self._last = time.time()
        self.motion_energy = 0.0
        self.last_vals: Optional[np.ndarray] = None
        self.last_residual: Optional[np.ndarray] = None
        self._bg_frames = 0

    def _vals(self, pkt: Dict[str, Any]) -> np.ndarray:
        raw = pkt.get("csi") or []
        try:
            v = np.array([float(x) for x in raw], dtype=float)
        except (TypeError, ValueError):
            v = np.zeros(0)
        if len(v) == 0:
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
        scores = []
        for k in ("movement_intensity", "activity"):
            if k in pkt and pkt[k] is not None:
                try:
                    scores.append(float(pkt[k]))
                except (TypeError, ValueError):
                    pass
        if self._prev is not None and len(self._prev) == len(vals):
            scores.append(float(np.mean(np.abs(vals - self._prev)) * 4.5))
        if not scores:
            scores.append(float(np.std(vals) * 2.0))
        return float(np.clip(max(scores), 0, 1))

    def _detections(self, vals: np.ndarray, motion: float) -> List[Detection]:
        # slower background lock-in after warmup
        if self._bg is None or len(self._bg) != len(vals):
            self._bg = vals.copy()
            self._bg_frames = 0
        else:
            a = 0.02 if self._bg_frames > 30 else 0.08
            self._bg = (1 - a) * self._bg + a * vals
            self._bg_frames += 1

        residual = np.clip(vals - self._bg, 0, None)
        if self._prev is not None and len(self._prev) == len(vals):
            # temporal change is the best body cue
            residual = 0.45 * residual + 0.55 * np.abs(vals - self._prev)
        residual = _smooth(residual, 3)
        rmax = float(residual.max()) + 1e-9
        residual_n = residual / rmax
        self.last_residual = residual_n.copy()
        self.last_vals = vals.copy()

        # need real motion before birthing bodies
        if motion < 0.07 and self._bg_frames > 20:
            return []

        dets: List[Detection] = []
        peaks = _peaks(residual_n, max_peaks=4)

        if motion > 0.12 and residual_n.mean() > 0.05:
            w = residual_n / (residual_n.sum() + 1e-9)
            idx = np.arange(len(residual_n))
            cx = float(0.12 + 0.76 * np.dot(w, idx) / max(1, len(residual_n) - 1))
            cy = float(0.22 + 0.50 * motion)
            feat = residual_n[:: max(1, len(residual_n) // 8)][:8].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            feat /= (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(cx, cy, motion, feat, prominence=float(residual_n.mean())))

        for i, strength in peaks:
            e = float(np.clip(0.3 * strength + 0.7 * motion, 0, 1))
            if e < 0.14 or strength < 0.12:
                continue
            x = float(0.10 + 0.80 * (i / max(1, len(residual_n) - 1)))
            y = float(0.20 + 0.55 * e)
            lo, hi = max(0, i - 3), min(len(residual_n), i + 4)
            feat = residual_n[lo:hi].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            else:
                feat = feat[:8]
            feat /= (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(x, y, e, feat, prominence=strength))

        return self._nms(dets, 0.14)

    def _nms(self, dets: List[Detection], min_dist: float) -> List[Detection]:
        dets = sorted(dets, key=lambda d: d.energy * (0.5 + d.prominence), reverse=True)
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
            used_d, used_t = set(), set()
            pairs = []
            for ti, tr in enumerate(self.tracks):
                tx, ty = tr.pos
                for di, d in enumerate(dets):
                    pd = math.hypot(tx - d.x, ty - d.y)
                    fd = float(np.linalg.norm(tr.feature - d.feature))
                    if pd > 0.30:
                        continue
                    pairs.append((1.0 * pd + 0.5 * fd, ti, di))
            pairs.sort()
            for cost, ti, di in pairs:
                if ti in used_t or di in used_d or cost > 0.48:
                    continue
                self.tracks[ti].update(dets[di])
                self.tracks[ti].rssi = rssi
                used_t.add(ti)
                used_d.add(di)

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                # stricter birth
                if d.energy < 0.16 or d.prominence < 0.08:
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
                if d.energy < 0.16 or d.prominence < 0.08:
                    continue
                if len(self.tracks) >= self.max_tracks:
                    break
                tr = KalmanTrack(d.x, d.y, d.energy, d.feature)
                tr.rssi = rssi
                self.tracks.append(tr)

        self._expire()

    def _drop_weakest(self):
        if self.tracks:
            w = min(self.tracks, key=lambda t: (t.confidence, t.hits, t.energy))
            self.tracks.remove(w)

    def _expire(self):
        now = time.time()
        self.tracks = [
            t for t in self.tracks
            if (now - t.last_seen) <= self.ttl_s
            and not (t.misses > 12 and t.confidence < 0.4)
            and not (t.state == "idle" and t.energy < 0.08 and t.misses > 6)
        ]

    def active(self) -> List[KalmanTrack]:
        self._expire()
        return sorted(
            [t for t in self.tracks if t.confidence > 0.30 and t.hits >= 3],
            key=lambda t: t.confidence * t.energy,
            reverse=True,
        )
