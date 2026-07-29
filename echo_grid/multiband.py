"""
Multiband / multi-node CSI fusion with intelligent overlap elimination.

Goal: one latent radio belief field. Band/node streams are different
observation operators. Dynamic mass is confirmed only when sources agree;
single-source peaks are treated as clutter or room structure, not tracks.

Commodity ESP nodes are often 2.4 GHz only — multi-NODE consistency uses
the same gate until true 5/6 GHz packets appear (band field in JSON).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def infer_band(pkt: Dict[str, Any]) -> str:
    """Normalize band label from packet metadata."""
    for key in ("band", "freq_band", "wifi_band"):
        v = pkt.get(key)
        if v is None:
            continue
        s = str(v).lower().strip()
        if "6" in s:
            return "6"
        if "5" in s:
            return "5"
        if "2.4" in s or "24" in s or s in ("2", "2g", "2.4g"):
            return "2.4"
    # channel number heuristics
    ch = pkt.get("channel") or pkt.get("primary_channel")
    try:
        ch = int(ch)
        if ch >= 1 and ch <= 14:
            return "2.4"
        if ch >= 36:
            return "5"
    except (TypeError, ValueError):
        pass
    # default commodity ESP CSI
    return "2.4"


# Relative trust / role weights for joint scoring
BAND_WEIGHT = {
    "2.4": 1.0,   # penetration / bulk structure
    "5": 1.25,    # spatial discrimination
    "6": 1.15,
    "unknown": 0.8,
}


@dataclass
class SourceObservation:
    node: str
    band: str
    motion: float
    residual: Optional[np.ndarray]
    vals: Optional[np.ndarray]
    rssi: float
    t: float
    addr: Any = None


@dataclass
class FusedDynamic:
    """Output of overlap-eliminated fusion for one epoch."""
    motion: float
    confidence: float
    n_sources: int
    n_bands: int
    agreed: bool
    occupancy: np.ndarray          # size x size dynamic mass [0,1]
    room: np.ndarray               # slow static belief [0,1]
    suppressed_single: int         # how many single-source peaks rejected


class RadioBeliefField:
    """Low-res persistent room + dynamic occupancy (tomography lite)."""

    def __init__(self, size: int = 16):
        self.size = size
        self.room = np.zeros((size, size), dtype=np.float32)      # static structure
        self.dynamic = np.zeros((size, size), dtype=np.float32)   # moving mass
        self.uncertainty = np.ones((size, size), dtype=np.float32)
        self._frames = 0

    def _project_residual(self, residual: np.ndarray, motion: float) -> np.ndarray:
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        if residual is None or len(residual) < 4:
            return grid
        n = len(residual)
        r = np.clip(residual.astype(np.float32), 0, None)
        r = r / (float(r.max()) + 1e-9)
        for k, v in enumerate(r):
            x = k / max(1, n - 1)
            ix = int(np.clip(x * (self.size - 1), 0, self.size - 1))
            cy = 0.30 + 0.50 * min(1.0, float(v) * max(0.2, motion))
            iy = int(np.clip(cy * (self.size - 1), 0, self.size - 1))
            amp = float(v) * max(0.15, motion)
            for j in range(self.size):
                for i in range(self.size):
                    d2 = (i - ix) ** 2 + (j - iy) ** 2 * 1.3 + 1e-6
                    grid[j, i] += amp * np.exp(-d2 * 0.30)
        m = float(grid.max()) + 1e-9
        return grid / m

    def update_from_sources(
        self,
        sources: List[SourceObservation],
        agreed_mask: Optional[np.ndarray] = None,
    ) -> FusedDynamic:
        self._frames += 1
        size = self.size
        if not sources:
            self.dynamic *= 0.92
            self.room *= 0.999
            return FusedDynamic(
                motion=0.0, confidence=0.0, n_sources=0, n_bands=0,
                agreed=False, occupancy=self.dynamic.copy(), room=self.room.copy(),
                suppressed_single=0,
            )

        # Per-source projected mass
        projections: List[Tuple[SourceObservation, np.ndarray]] = []
        for src in sources:
            res = src.residual if src.residual is not None else src.vals
            if res is None:
                continue
            proj = self._project_residual(np.asarray(res, dtype=float), src.motion)
            w = BAND_WEIGHT.get(src.band, 0.8) * (0.5 + 0.5 * src.motion)
            projections.append((src, proj * w))

        if not projections:
            self.dynamic *= 0.92
            return FusedDynamic(
                motion=float(np.mean([s.motion for s in sources])),
                confidence=0.1, n_sources=len(sources),
                n_bands=len({s.band for s in sources}),
                agreed=False, occupancy=self.dynamic.copy(), room=self.room.copy(),
                suppressed_single=0,
            )

        stack = np.stack([p for _, p in projections], axis=0)
        # Agreement map: geometric mean of active sources (overlap)
        eps = 1e-6
        agree = np.exp(np.mean(np.log(stack + eps), axis=0))
        agree = agree / (float(agree.max()) + 1e-9)

        # Union mass (any source)
        union = np.max(stack, axis=0)
        union = union / (float(union.max()) + 1e-9)

        n_src = len(projections)
        n_bands = len({s.band for s, _ in projections})

        # Intelligent overlap elimination:
        # - multi-source or multi-band: trust agreement region
        # - single source: only strong motion, heavily damped (clutter quarantine)
        suppressed = 0
        if n_src >= 2 or n_bands >= 2:
            dynamic_new = agree * np.clip(union, 0, 1)
            # boost where at least 2 sources have mass
            support = (stack > 0.15).sum(axis=0)
            dynamic_new = dynamic_new * np.clip(support / max(1, n_src - 1), 0.35, 1.5)
            agreed = True
            conf = float(np.clip(0.35 + 0.25 * n_src + 0.2 * n_bands, 0, 0.95))
        else:
            # single stream: allow weak dynamic but mark low confidence
            dynamic_new = union * 0.45
            agreed = False
            conf = float(np.clip(0.15 + 0.4 * sources[0].motion, 0, 0.55))
            suppressed = int(np.sum(union > 0.4))

        if agreed_mask is not None and agreed_mask.shape == dynamic_new.shape:
            dynamic_new = dynamic_new * agreed_mask

        # EMA into persistent fields
        a_dyn = 0.35 if agreed else 0.18
        self.dynamic = (1 - a_dyn) * self.dynamic + a_dyn * dynamic_new.astype(np.float32)

        # Room memory: slow uptake of mass that is stable & low-motion
        mean_motion = float(np.mean([s.motion for s in sources]))
        if mean_motion < 0.12:
            a_room = 0.02
            self.room = (1 - a_room) * self.room + a_room * union.astype(np.float32)
        else:
            # while moving, slightly forget room under dynamic peaks
            self.room = np.clip(self.room - 0.01 * self.dynamic, 0, 1)

        self.uncertainty = np.clip(
            0.97 * self.uncertainty + 0.03 * (1.0 - self.dynamic),
            0.05, 1.0,
        ).astype(np.float32)

        # Decay dynamic when quiet
        if mean_motion < 0.06:
            self.dynamic *= 0.94

        motion_out = float(np.clip(
            mean_motion * (1.15 if agreed else 0.7) * (0.6 + 0.4 * float(self.dynamic.max())),
            0, 1,
        ))

        return FusedDynamic(
            motion=motion_out,
            confidence=conf,
            n_sources=n_src,
            n_bands=n_bands,
            agreed=agreed,
            occupancy=self.dynamic.copy(),
            room=self.room.copy(),
            suppressed_single=suppressed,
        )

    def spatial_for_display(self) -> np.ndarray:
        """Blend dynamic over room for UI (dynamic dominates)."""
        out = np.clip(0.25 * self.room + 0.95 * self.dynamic, 0, 1)
        m = float(out.max()) + 1e-9
        return (out / m).astype(np.float32)


class MultibandFuser:
    """Registry of recent source observations + belief field update."""

    def __init__(self, size: int = 16, source_ttl: float = 1.2):
        self.field = RadioBeliefField(size=size)
        self.source_ttl = source_ttl
        self.sources: Dict[str, SourceObservation] = {}
        self.last_fused: Optional[FusedDynamic] = None
        self.gate_log = 0

    def _source_key(self, node: str, band: str) -> str:
        return f"{node}|{band}"

    def observe(
        self,
        node: str,
        pkt: Dict[str, Any],
        motion: float,
        residual: Optional[np.ndarray],
        vals: Optional[np.ndarray],
        addr: Any = None,
    ) -> FusedDynamic:
        band = infer_band(pkt)
        try:
            rssi = float(pkt.get("rssi", -90))
        except (TypeError, ValueError):
            rssi = -90.0

        key = self._source_key(node, band)
        self.sources[key] = SourceObservation(
            node=node,
            band=band,
            motion=float(np.clip(motion, 0, 1)),
            residual=None if residual is None else np.asarray(residual, dtype=float),
            vals=None if vals is None else np.asarray(vals, dtype=float),
            rssi=rssi,
            t=time.time(),
            addr=addr,
        )

        # expire stale
        now = time.time()
        self.sources = {
            k: v for k, v in self.sources.items()
            if now - v.t <= self.source_ttl
        }

        active = list(self.sources.values())
        fused = self.field.update_from_sources(active)
        self.last_fused = fused

        if self.gate_log < 12 and (fused.agreed or fused.n_sources >= 2):
            print(
                f"[multiband] sources={fused.n_sources} bands={fused.n_bands} "
                f"agreed={fused.agreed} motion={fused.motion:.3f} conf={fused.confidence:.2f}"
            )
            self.gate_log += 1
        return fused

    def should_trust_tracks(self) -> bool:
        """Track birth from raw peaks only if fusion agrees or single strong source."""
        f = self.last_fused
        if f is None:
            return True  # cold start
        if f.agreed:
            return True
        # single source: allow tracks only at higher motion
        return f.motion > 0.22 and f.confidence > 0.3

    def fused_motion(self) -> float:
        if self.last_fused is None:
            return 0.0
        return float(self.last_fused.motion)

    def fused_confidence(self) -> float:
        if self.last_fused is None:
            return 0.0
        return float(self.last_fused.confidence)
