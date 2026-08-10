"""
Live motion head for Echo Grid.

Loads the MetaField echo_head.pt checkpoint (TinyFieldHead) and scores
motion residual online so the dashboard can display learned surprise.

Optional torch dependency — only needed to load the checkpoint once.
Forward pass is pure numpy after load.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES = [
    "motion", "entropy", "df_max", "drive", "fuse",
    "n_tracks", "track_energy", "track_conf",
]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def features_from_osys(osys: Any) -> List[float]:
    """Match MetaField echo_field_predictor feature layout."""
    motion = _clip01(float(getattr(osys, "last_csi_energy", 0.0)))
    entropy = float(getattr(osys.field, "entropy", 0.0))
    entropy_n = _clip01(entropy / 1.5)
    df_max = float(getattr(osys, "last_df_max", 0.0))
    df_n = _clip01(df_max / 2000.0)
    drive = _clip01(float(getattr(osys.field, "_drive", 0.0)))
    fuse = _clip01(float(getattr(osys, "fuse_conf", 0.0)))

    tracks = []
    if getattr(osys, "csi", None) is not None:
        try:
            tracks = list(osys.csi.active_tracks())
        except Exception:
            tracks = []

    n_tracks = float(len(tracks))
    if tracks:
        te = sum(_clip01(float(getattr(t, "energy", 0.0))) for t in tracks) / len(tracks)
        tc = sum(_clip01(float(getattr(t, "confidence", 0.5))) for t in tracks) / len(tracks)
    else:
        te, tc = 0.0, 0.0

    return [motion, entropy_n, df_n, drive, fuse, n_tracks / 6.0, te, tc]


class _NumpyMLP:
    """TinyFieldHead forward in numpy: Linear-ReLU-Linear-ReLU-Linear-Sigmoid."""

    def __init__(self, weights: List[Tuple[np.ndarray, np.ndarray]]):
        # weights: list of (W [out,in], b [out]) for each Linear
        self.layers = weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)  # ReLU
            else:
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -40, 40)))  # Sigmoid
        return h.astype(np.float32)


def _load_torch_mlp(path: Path) -> Tuple[_NumpyMLP, int, int]:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required once to load echo_head.pt — pip install torch"
        ) from e

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    window = int(ckpt.get("window", 8))
    in_dim = int(ckpt.get("in_dim", window * len(FEATURE_NAMES)))

    # net.0, net.2, net.4 are Linear layers in TinyFieldHead
    layers = []
    for idx in (0, 2, 4):
        W = sd[f"net.{idx}.weight"].detach().cpu().numpy()
        b = sd[f"net.{idx}.bias"].detach().cpu().numpy()
        layers.append((W, b))
    return _NumpyMLP(layers), window, in_dim


class LiveFieldHead:
    """Rolling-window residual scorer for EchoGridOS."""

    def __init__(self, model_path: str | Path, threshold: float = 0.30):
        self.path = Path(model_path)
        self.threshold = float(threshold)
        self.mlp, self.window, self.in_dim = _load_torch_mlp(self.path)
        self.history: Deque[List[float]] = deque(maxlen=self.window)
        self.last_pred_motion = 0.0
        self.last_actual_motion = 0.0
        self.last_residual = 0.0
        self.last_abs_residual = 0.0
        self.surprise = False
        self.ready = False
        self.n_scored = 0
        self.n_surprise = 0
        print(
            f"[field_head] loaded {self.path}  window={self.window}  "
            f"in_dim={self.in_dim}  threshold={self.threshold:.2f}"
        )

    def update(self, osys: Any) -> Dict[str, float]:
        feats = features_from_osys(osys)
        if len(self.history) < self.window:
            self.history.append(feats)
            self.ready = False
            self.surprise = False
            return {
                "pred_motion": 0.0,
                "actual_motion": feats[0],
                "residual": 0.0,
                "abs_residual": 0.0,
                "surprise": 0.0,
                "ready": 0.0,
            }

        x = np.array([v for row in self.history for v in row], dtype=np.float64)
        if x.shape[0] != self.in_dim:
            # window/feature mismatch — reset
            self.history.clear()
            self.history.append(feats)
            self.ready = False
            return {
                "pred_motion": 0.0,
                "actual_motion": feats[0],
                "residual": 0.0,
                "abs_residual": 0.0,
                "surprise": 0.0,
                "ready": 0.0,
            }

        pred = self.mlp(x)
        pred_m = float(pred[0])
        actual_m = float(feats[0])
        residual = actual_m - pred_m
        abs_r = abs(residual)

        self.history.append(feats)
        self.last_pred_motion = pred_m
        self.last_actual_motion = actual_m
        self.last_residual = residual
        self.last_abs_residual = abs_r
        self.surprise = abs_r >= self.threshold
        self.ready = True
        self.n_scored += 1
        if self.surprise:
            self.n_surprise += 1

        return {
            "pred_motion": pred_m,
            "actual_motion": actual_m,
            "residual": residual,
            "abs_residual": abs_r,
            "surprise": 1.0 if self.surprise else 0.0,
            "ready": 1.0,
        }
