"""
Echo Grid → MetaField FieldObservation bridge.

Emits the same JSON contract as optical-body-s3 / schemas/field_observation.py
so MetaField can ingest Echo as just another physical body.

No dependency on the metafield package — pure dict/JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def build_observation(
    osys: Any,
    body_id: str = "echo-grid-01",
    excitation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Snapshot EchoGridOS state into a FieldObservation-shaped dict.

    Regions (stable names):
      motion, entropy, df_max, drive,
      track_<id> for each active track
    """
    motion = _clip01(float(getattr(osys, "last_csi_energy", 0.0)))
    entropy = float(getattr(osys.field, "entropy", 0.0))
    entropy_n = _clip01(entropy / 1.5)
    df_max = float(getattr(osys, "last_df_max", 0.0))
    df_n = _clip01(df_max / 2000.0)
    drive = _clip01(float(getattr(osys.field, "_drive", 0.0)))

    fuse_agreed = bool(getattr(osys, "fuse_agreed", False))
    fuse_conf = _clip01(float(getattr(osys, "fuse_conf", 0.0)))
    fuse_sources = int(getattr(osys, "fuse_sources", 0))
    fuse_bands = int(getattr(osys, "fuse_bands", 0))
    packets = int(getattr(osys, "csi_packets", 0))

    regions: List[Dict[str, Any]] = [
        {
            "region": "motion",
            "observed": motion,
            "expected": None,
            "confidence": 0.9 if packets > 5 else 0.5,
            "anomaly": 0.0,
        },
        {
            "region": "entropy",
            "observed": entropy_n,
            "expected": None,
            "confidence": 0.85,
            "anomaly": 0.0,
            "extras": {"raw_entropy": entropy},
        },
        {
            "region": "df_max",
            "observed": df_n,
            "expected": None,
            "confidence": 0.8,
            "anomaly": 0.0,
            "extras": {"raw_hz": df_max},
        },
        {
            "region": "drive",
            "observed": drive,
            "expected": None,
            "confidence": 0.9,
            "anomaly": 0.0,
        },
        {
            "region": "fuse",
            "observed": fuse_conf,
            "expected": None,
            "confidence": fuse_conf,
            "anomaly": 0.0 if fuse_agreed else 0.35,
            "extras": {
                "agreed": fuse_agreed,
                "sources": fuse_sources,
                "bands": fuse_bands,
            },
        },
    ]

    tracks = []
    if getattr(osys, "csi", None) is not None:
        try:
            tracks = list(osys.csi.active_tracks())
        except Exception:
            tracks = []

    for tr in tracks:
        try:
            x, y = tr.pos
            conf = _clip01(float(tr.confidence))
            energy = _clip01(float(tr.energy))
            tid = str(getattr(tr, "track_id", "T?"))
            state = str(getattr(tr, "state", "unknown"))
            regions.append({
                "region": f"track_{tid}",
                "observed": energy,
                "expected": None,
                "confidence": conf,
                "anomaly": 0.0,
                "extras": {
                    "x": float(x),
                    "y": float(y),
                    "state": state,
                    "speed": float(getattr(tr, "speed", 0.0)),
                },
            })
        except Exception:
            continue

    health = "ok"
    if packets == 0 and getattr(osys, "csi_enabled", False):
        health = "stale"
    elif motion > 0.95 and not fuse_agreed and fuse_sources < 2:
        health = "partial"

    return {
        "schema_version": 1,
        "body_id": body_id,
        "body_type": "ultrasonic",
        "excitation_id": excitation_id,
        "field_regions": regions,
        "geometry_state": "calibrated" if packets > 20 else "uncalibrated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modality": {
            "echo": {
                "t": float(getattr(osys, "t", 0.0)),
                "csi_packets": packets,
                "n_tracks": len(tracks),
                "fuse_agreed": fuse_agreed,
                "fuse_sources": fuse_sources,
                "fuse_bands": fuse_bands,
                "fuse_conf": fuse_conf,
                "entropy_raw": entropy,
                "df_max_hz": df_max,
                "drive": drive,
                "motion": motion,
            }
        },
        "health": health,
    }


class MetaFieldEmitter:
    """Append FieldObservation JSON lines for MetaField consumption."""

    def __init__(
        self,
        path: Path | str,
        body_id: str = "echo-grid-01",
        every_n: int = 4,
    ):
        self.path = Path(path)
        self.body_id = body_id
        self.every_n = max(1, int(every_n))
        self._frame = 0
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty file immediately so MetaField --follow unblocks
        if not self.path.exists():
            self.path.touch()
        print(f"[metafield] emitter ready → {self.path.resolve()}")

    def maybe_emit(self, osys: Any) -> Optional[Dict[str, Any]]:
        self._frame += 1
        if self._frame % self.every_n != 0:
            return None
        return self.emit_now(osys)

    def emit_now(self, osys: Any) -> Dict[str, Any]:
        obs = build_observation(osys, body_id=self.body_id, excitation_id=self._count)
        self._count += 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs, default=str) + "\n")
        return obs
