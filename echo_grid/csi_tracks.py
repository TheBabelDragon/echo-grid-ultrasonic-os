"""CSI body tracking — adaptive residual ML + confirmed multi-target tracks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _smooth(vals: np.ndarray, k: int = 5) -> np.ndarray:
    if len(vals) < k:
        return vals.copy()
    pad = k // 2
    xp = np.pad(vals, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _zscore_peaks(z: np.ndarray, max_peaks: int = 5) -> List[Tuple[int, float]]:
    n = len(z)
    if n < 5:
        return []
    s = _smooth(z, 5)
    out = []
    for i in range(2, n - 2):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > 1.2:
            prom = s[i] - 0.5 * (s[i - 1] + s[i + 1])
            if prom > 0.35:
                out.append((i, float(s[i] * (1.0 + 0.5 * prom))))
    out.sort(key=lambda t: t[1], reverse=True)
    # NMS in index space
    kept: List[Tuple[int, float]] = []
    for i, sc in out:
        if all(abs(i - j) >= 3 for j, _ in kept):
            kept.append((i, sc))
        if len(kept) >= max_peaks:
            break
    return kept


@dataclass
class Detection:
    x: float
    y: float
    energy: float
    feature: np.ndarray
    z: float = 0.0


class KalmanTrack:
    _id = 1

    def __init__(self, x: float, y: float, energy: float, feature: np.ndarray):
        self.track_id = f"T{KalmanTrack._id}"
        KalmanTrack._id += 1
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 0.2
        self.energy = float(energy)
        self.feature = feature.astype(float).copy()
        self.feature /= (np.linalg.norm(self.feature) + 1e-9)
        self.confidence = 0.25
        self.hits = 1
        self.misses = 0
        self.last_seen = time.time()
        self.state = "tentative"  # tentative | idle | move | surge
        self.confirmed = False
        self.rssi = -90.0
        self.age = 0
        self.energy_hist: List[float] = [float(energy)]

    @property
    def pos(self) -> Tuple[float, float]:
        return float(np.clip(self.x[0], 0, 1)), float(np.clip(self.x[1], 0, 1))

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    def predict(self, dt: float):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = 0.03 + 0.15 * (1.0 - self.confidence)
        Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ], dtype=float)
        self.x = F @ self.x
        self.x[0] = float(np.clip(self.x[0], 0, 1))
        self.x[1] = float(np.clip(self.x[1], 0, 1))
        self.P = F @ self.P @ F.T + Q
        self.misses += 1
        self.age += 1
        self.confidence *= 0.975
        self.energy *= 0.985

    def update(self, det: Detection):
        z = np.array([det.x, det.y], dtype=float)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * (0.02 + 0.07 * (1.0 - min(1.0, det.energy)))
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

        self.energy = 0.5 * self.energy + 0.5 * det.energy
        self.energy_hist.append(self.energy)
        if len(self.energy_hist) > 30:
            self.energy_hist = self.energy_hist[-30:]

        self.feature = 0.65 * self.feature + 0.35 * det.feature
        self.feature /= (np.linalg.norm(self.feature) + 1e-9)

        self.hits += 1
        self.misses = 0
        self.last_seen = time.time()
        self.age += 1

        if self.hits >= 4:
            self.confirmed = True

        unc = float(self.P[0, 0] + self.P[1, 1])
        e_std = float(np.std(self.energy_hist)) if len(self.energy_hist) > 3 else 0.1
        stab = 1.0 / (1.0 + self.speed + e_std)
        self.confidence = float(np.clip(
            0.10
            + 0.35 * self.energy
            + 0.25 * min(self.hits, 20) / 20
            + 0.15 * stab
            + 0.15 / (1 + 8 * unc),
            0.05, 0.99,
        ))

        sp = self.speed
        if not self.confirmed:
            self.state = "tentative"
        elif self.energy > 0.55 and sp > 0.08:
            self.state = "surge"
        elif sp > 0.04 or self.energy > 0.28:
            self.state = "move"
        else:
            self.state = "idle"


class TrackStore:
    def __init__(self, max_tracks: int = 6, ttl_s: float = 2.4):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: List[KalmanTrack] = []
        self._bg_mean: Optional[np.ndarray] = None
        self._bg_var: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self._motion_ema = 0.0
        self._last = time.time()
        self.motion_energy = 0.0
        self.last_vals: Optional[np.ndarray] = None
        self.last_residual: Optional[np.ndarray] = None
        self._frames = 0

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
            scores.append(float(np.mean(np.abs(vals - self._prev)) * 5.0))
        if self._bg_mean is not None and len(self._bg_mean) == len(vals):
            scores.append(float(np.mean(np.abs(vals - self._bg_mean)) * 3.0))
        if not scores:
            scores.append(float(np.std(vals) * 2.5))
        m = float(np.clip(max(scores), 0, 1))
        self._motion_ema = 0.7 * self._motion_ema + 0.3 * m
        return float(np.clip(0.5 * m + 0.5 * self._motion_ema, 0, 1))

    def _update_background(self, vals: np.ndarray, motion: float):
        if self._bg_mean is None or len(self._bg_mean) != len(vals):
            self._bg_mean = vals.copy()
            self._bg_var = np.ones_like(vals) * 0.05
            return
        # slower update when motion high (don't absorb bodies into background)
        a = 0.01 if motion > 0.25 else (0.04 if self._frames > 40 else 0.1)
        delta = vals - self._bg_mean
        self._bg_mean = (1 - a) * self._bg_mean + a * vals
        self._bg_var = (1 - a) * self._bg_var + a * (delta * delta)
        self._bg_var = np.maximum(self._bg_var, 1e-4)

    def _detections(self, vals: np.ndarray, motion: float) -> List[Detection]:
        self._update_background(vals, motion)
        assert self._bg_mean is not None and self._bg_var is not None

        residual = vals - self._bg_mean
        z = residual / (np.sqrt(self._bg_var) + 1e-6)
        if self._prev is not None and len(self._prev) == len(vals):
            temp = (vals - self._prev) / (np.sqrt(self._bg_var) + 1e-6)
            z = 0.4 * z + 0.6 * temp

        z_pos = np.clip(z, 0, None)
        z_n = z_pos / (z_pos.max() + 1e-9)
        self.last_residual = z_n.copy()
        self.last_vals = vals.copy()

        if motion < 0.06 and self._frames > 25:
            return []

        dets: List[Detection] = []
        peaks = _zscore_peaks(z_pos, max_peaks=5)

        # global centroid detection for diffuse motion
        if motion > 0.12 and float(z_n.mean()) > 0.04:
            w = z_n / (z_n.sum() + 1e-9)
            idx = np.arange(len(z_n))
            cx = float(0.12 + 0.76 * np.dot(w, idx) / max(1, len(z_n) - 1))
            cy = float(0.22 + 0.50 * motion)
            feat = z_n[:: max(1, len(z_n) // 8)][:8].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            feat /= (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(cx, cy, motion, feat, z=float(z_pos.mean())))

        for i, strength in peaks:
            e = float(np.clip(0.25 * min(strength / 4.0, 1.0) + 0.75 * motion, 0, 1))
            if e < 0.12 or strength < 1.3:
                continue
            x = float(0.10 + 0.80 * (i / max(1, len(z_n) - 1)))
            y = float(0.20 + 0.55 * e)
            lo, hi = max(0, i - 3), min(len(z_n), i + 4)
            feat = z_n[lo:hi].astype(float)
            if len(feat) < 8:
                feat = np.pad(feat, (0, 8 - len(feat)))
            else:
                feat = feat[:8]
            feat /= (np.linalg.norm(feat) + 1e-9)
            dets.append(Detection(x, y, e, feat, z=strength))

        return self._nms(dets, 0.12)

    def _nms(self, dets: List[Detection], min_dist: float) -> List[Detection]:
        dets = sorted(dets, key=lambda d: d.energy * (0.5 + 0.1 * d.z), reverse=True)
        kept: List[Detection] = []
        for d in dets:
            if all(math.hypot(d.x - k.x, d.y - k.y) >= min_dist for k in kept):
                kept.append(d)
        return kept

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        now = time.time()
        dt = float(np.clip(now - self._last, 1e-3, 0.35))
        self._last = now
        self._frames += 1

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
                    # velocity gate: prefer detections ahead of motion
                    pred_x = tx + tr.x[2] * dt * 2
                    pred_y = ty + tr.x[3] * dt * 2
                    pd_v = math.hypot(pred_x - d.x, pred_y - d.y)
                    fd = 1.0 - float(np.dot(tr.feature, d.feature))
                    if pd > 0.32 and pd_v > 0.35:
                        continue
                    cost = 0.55 * min(pd, pd_v) + 0.45 * fd
                    pairs.append((cost, ti, di))
            pairs.sort()
            for cost, ti, di in pairs:
                if ti in used_t or di in used_d or cost > 0.55:
                    continue
                self.tracks[ti].update(dets[di])
                self.tracks[ti].rssi = rssi
                used_t.add(ti)
                used_d.add(di)

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                if d.energy < 0.15 or d.z < 1.0:
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
                if d.energy < 0.15 or d.z < 1.0:
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
        # prefer dropping unconfirmed
        pool = [t for t in self.tracks if not t.confirmed] or self.tracks
        w = min(pool, key=lambda t: (t.confidence, t.hits, t.energy))
        self.tracks.remove(w)

    def _expire(self):
        now = time.time()
        alive = []
        for t in self.tracks:
            if (now - t.last_seen) > self.ttl_s:
                continue
            if t.misses > 14 and t.confidence < 0.35:
                continue
            if not t.confirmed and t.misses > 5:
                continue
            if t.state == "idle" and t.energy < 0.07 and t.misses > 8:
                continue
            alive.append(t)
        self.tracks = alive

    def active(self) -> List[KalmanTrack]:
        self._expire()
        # confirmed preferred; high-confidence tentative allowed briefly
        out = [
            t for t in self.tracks
            if (t.confirmed and t.confidence > 0.28)
            or (not t.confirmed and t.hits >= 3 and t.confidence > 0.45)
        ]
        return sorted(out, key=lambda t: t.confidence * t.energy, reverse=True)
