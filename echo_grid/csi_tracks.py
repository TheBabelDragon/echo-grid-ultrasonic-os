"""
Multi-body CSI track layer for Echo Grid.

Practical scope (v1):
  - One track per CSI node_id (multi-ESP)
  - Extra local peaks from a single node's subcarrier profile
  - Smoothed (x, y, energy) for display circles + field injection

Not yet: full ML person re-ID / through-wall identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: str
    x: float = 0.5
    y: float = 0.5
    energy: float = 0.0
    rssi: float = -90.0
    last_seen: float = field(default_factory=time.time)
    hits: int = 0

    def smooth_toward(self, x: float, y: float, energy: float, alpha: float = 0.35):
        self.x = (1 - alpha) * self.x + alpha * x
        self.y = (1 - alpha) * self.y + alpha * y
        self.energy = (1 - alpha) * self.energy + alpha * energy
        self.last_seen = time.time()
        self.hits += 1


class TrackStore:
    def __init__(self, max_tracks: int = 6, ttl_s: float = 3.0):
        self.max_tracks = max_tracks
        self.ttl_s = ttl_s
        self.tracks: Dict[str, Track] = {}

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

    def _position_from_csi(self, pkt: Dict[str, Any], energy: float) -> Tuple[float, float]:
        csi = pkt.get("csi") or []
        try:
            vals = [float(v) for v in csi]
        except (TypeError, ValueError):
            vals = []

        if len(vals) >= 8:
            mid = len(vals) // 2
            left = sum(vals[:mid]) / mid
            right = sum(vals[mid:]) / max(1, len(vals) - mid)
            total = left + right + 1e-6
            x = 0.25 + 0.50 * (right / total)
            y = 0.30 + 0.40 * energy
        else:
            x, y = 0.5, 0.5
        return (x, y)

    def _peak_tracks(self, pkt: Dict[str, Any], node: str) -> List[Tuple[str, float, float, float]]:
        """Split a single CSI vector into up to 2 spatial peaks."""
        csi = pkt.get("csi") or []
        try:
            vals = [float(v) for v in csi]
        except (TypeError, ValueError):
            return []

        if len(vals) < 12:
            return []

        n = len(vals)
        third = n // 3
        regions = [
            ("L", vals[:third], 0.25),
            ("C", vals[third:2 * third], 0.50),
            ("R", vals[2 * third:], 0.75),
        ]
        scored = []
        for name, chunk, x in regions:
            if not chunk:
                continue
            e = sum(chunk) / len(chunk)
            scored.append((name, x, e))
        scored.sort(key=lambda t: t[2], reverse=True)

        out = []
        for name, x, e in scored[:2]:
            if e < 0.12:
                continue
            y = 0.35 + 0.35 * min(1.0, e)
            out.append((f"{node}:{name}", x, y, float(min(1.0, e))))
        return out

    def update_from_packet(self, pkt: Dict[str, Any]) -> None:
        node = str(pkt.get("node") or pkt.get("body_id") or "csi_unknown")
        energy = self._energy(pkt)
        try:
            rssi = float(pkt.get("rssi", -90))
        except (TypeError, ValueError):
            rssi = -90.0

        # Primary track = whole node
        x, y = self._position_from_csi(pkt, energy)
        tid = node
        if tid not in self.tracks:
            if len(self.tracks) >= self.max_tracks:
                self._drop_oldest()
            self.tracks[tid] = Track(track_id=tid, x=x, y=y, energy=energy, rssi=rssi)
        self.tracks[tid].smooth_toward(x, y, energy)
        self.tracks[tid].rssi = rssi

        # Optional local peaks as extra circles when energy is rich
        if energy > 0.25:
            for pid, px, py, pe in self._peak_tracks(pkt, node):
                if pid not in self.tracks:
                    if len(self.tracks) >= self.max_tracks:
                        break
                    self.tracks[pid] = Track(track_id=pid, x=px, y=py, energy=pe, rssi=rssi)
                self.tracks[pid].smooth_toward(px, py, pe)

        self._expire()

    def _drop_oldest(self):
        if not self.tracks:
            return
        oldest = min(self.tracks.values(), key=lambda t: t.last_seen)
        del self.tracks[oldest.track_id]

    def _expire(self):
        now = time.time()
        dead = [k for k, t in self.tracks.items() if now - t.last_seen > self.ttl_s]
        for k in dead:
            del self.tracks[k]

    def active(self) -> List[Track]:
        self._expire()
        return sorted(self.tracks.values(), key=lambda t: t.energy, reverse=True)
