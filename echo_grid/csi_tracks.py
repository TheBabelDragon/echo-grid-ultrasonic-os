"""
Multi-body CSI tracks — improved spatial intelligence.

- Prominence-based subcarrier peak detection
- Constant-velocity smoothing
- Confidence from stability + evidence
- Greedy association to reduce ID flicker
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: str
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    energy: float = 0.0
    confidence: float = 0.3
    rssi: float = -90.0
    last_seen: float = field(default_factory=time.time)
    hits: int = 0

    def predict(self, dt: float):
        self.x = min(1.0, max(0.0, self.x + self.vx * dt))
        self.y = min(1.0, max(0.0, self.y + self.vy * dt))

    def update(self, x: float, y: float, energy: float, alpha: float = 0.4):
        now = time.time()
        dt = max(1e-3, now - self.last_seen)
        # measured residual
        mx, my = x - self.x, y - self.y
        # velocity estimate
        self.vx = (1 - alpha) * self.vx + alpha * (mx / dt)
        self.vy = (1 - alpha) * self.vy + alpha * (my / dt)
        # clamp speed (normalized field units / sec)
        speed = math.hypot(self.vx, self.vy)
        if speed > 1.5:
            self.vx *= 1.5 / speed
            self.vy *= 1.5 / speed
        self.x = (1 - alpha) * self.x + alpha * x
        self.y = (1 - alpha) * self.y + alpha * y
        self.energy = (1 - alpha) * self.energy + alpha * energy
        self.hits += 1
        self.last_seen = now
        # confidence rises with hits and energy, falls with jittery motion
        stab = 1.0 / (1.0 + speed)
        self.confidence = min(0.98, 0.25 + 0.08 * min(self.hits, 8) + 0.4 * self.energy * stab)


def _find_peaks(vals: List[float], max_peaks: int = 3) -> List[Tuple[int, float]]:
    """Return (index, prominence) peaks."""
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
            prom = vals[i] - max(left, right)
            if prom > 0.04 and vals[i] > 0.12:
                peaks.append((i, prom * vals[i]))
    peaks.sort(key=lambda t: t[1], reverse=True)
    return peaks[:max_peaks]


class TrackStore:
    def __init__(self, max_tracks: int = 6, ttl_s: float = 2.5):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: Dict[str, Track] = {}
        self._last_update = time.time()

    def _energy(self, pkt: Dict[str, Any]) -> float:
        for key in ("movement_intensity", "activity"):
            if key in pkt and pkt[key] is not None:
                try:
                    return float(min(1.0, max(0.0, float(pkt[key]))))
                except (TypeError, ValueError):
                    pass
        csi = pkt.get("csi") or []
        try:
            vals = [float(v) for v in csi]
        except (TypeError, ValueError):
            return 0.0
        if not vals:
            return 0.0
        return float(min(1.0, sum(vals) / len(vals)))

    def _detections(self, pkt: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
        """List of (suggested_id, x, y, energy)."""
        node = str(pkt.get("node") or "csi")
        energy = self._energy(pkt)
        csi = pkt.get("csi") or []
        try:
            vals = [float(v) for v in csi]
        except (TypeError, ValueError):
            vals = []

        dets: List[Tuple[str, float, float, float]] = []

        # Always one whole-node detection
        if len(vals) >= 4:
            mid = len(vals) // 2
            left = sum(vals[:mid]) / max(1, mid)
            right = sum(vals[mid:]) / max(1, len(vals) - mid)
            total = left + right + 1e-6
            x = 0.22 + 0.56 * (right / total)
            y = 0.28 + 0.45 * energy
        else:
            x, y = 0.5, 0.45
        dets.append((node, x, y, energy))

        # Peak-based extra bodies
        if len(vals) >= 12 and energy > 0.15:
            peaks = _find_peaks(vals, max_peaks=3)
            for rank, (idx, score) in enumerate(peaks):
                e = float(min(1.0, vals[idx]))
                if e < 0.15:
                    continue
                px = 0.15 + 0.70 * (idx / max(1, len(vals) - 1))
                py = 0.25 + 0.50 * e
                dets.append((f"{node}:p{rank}", px, py, e))

        return dets

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        now = time.time()
        dt = max(1e-3, now - self._last_update)
        self._last_update = now

        for tr in self.tracks.values():
            tr.predict(dt)

        dets = self._detections(pkt)
        try:
            rssi = float(pkt.get("rssi", -90))
        except (TypeError, ValueError):
            rssi = -90.0

        unmatched_dets = list(range(len(dets)))
        unmatched_tracks = list(self.tracks.keys())

        # Greedy nearest association in (x,y)
        pairs = []
        for ti, tid in enumerate(unmatched_tracks):
            tr = self.tracks[tid]
            for di in unmatched_dets:
                _, x, y, e = dets[di]
                dist = math.hypot(tr.x - x, tr.y - y)
                if dist < 0.28:
                    pairs.append((dist, tid, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            _, x, y, e = dets[di]
            self.tracks[tid].update(x, y, e)
            self.tracks[tid].rssi = rssi
            used_t.add(tid)
            used_d.add(di)

        # New tracks for unmatched detections
        for di, det in enumerate(dets):
            if di in used_d:
                continue
            sid, x, y, e = det
            if e < 0.08:
                continue
            # Prefer stable id; if collision, unique suffix
            tid = sid
            if tid in self.tracks and tid not in used_t:
                # existing track far away — new id
                tid = f"{sid}_{int(now * 10) % 1000}"
            if tid not in self.tracks:
                if len(self.tracks) >= self.max_tracks:
                    self._drop_weakest()
                if len(self.tracks) >= self.max_tracks:
                    continue
                self.tracks[tid] = Track(track_id=tid, x=x, y=y, energy=e, rssi=rssi)
            self.tracks[tid].update(x, y, e)
            self.tracks[tid].rssi = rssi

        self._expire()

    def _drop_weakest(self):
        if not self.tracks:
            return
        weak = min(self.tracks.values(), key=lambda t: (t.confidence, t.energy, t.hits))
        del self.tracks[weak.track_id]

    def _expire(self):
        now = time.time()
        dead = [k for k, t in self.tracks.items() if now - t.last_seen > self.ttl_s]
        for k in dead:
            del self.tracks[k]

    def active(self) -> List[Track]:
        self._expire()
        # require minimal confidence for display
        out = [t for t in self.tracks.values() if t.confidence > 0.28 or t.hits >= 2]
        return sorted(out, key=lambda t: t.energy * t.confidence, reverse=True)
