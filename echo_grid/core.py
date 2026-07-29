"""Echo Grid — fused radio belief → φ inject (multiband-aware)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


class EchoFieldOS:
    def __init__(self, size: int = 16):
        self.size = size
        self.phi = np.zeros((size, size), dtype=np.float32)
        self.vel = np.zeros((size, size), dtype=np.float32)
        self.lmbda = 0.12
        self.gamma = 0.94
        self.max_abs = 2.5
        self.entropy = 0.0
        self._drive = 0.0

    def inject(self, x: float, y: float, force: float = 1.0):
        ix = int(np.clip(x * self.size, 0, self.size - 1))
        iy = int(np.clip(y * self.size, 0, self.size - 1))
        force = float(np.clip(force, -1.8, 1.8))
        self._drive = min(1.0, 0.75 * self._drive + 0.45 * abs(force))
        for j in range(self.size):
            for i in range(self.size):
                dx, dy = i - ix, j - iy
                d2 = dx * dx + dy * dy + 1e-6
                self.vel[j, i] += np.exp(-d2 * 0.12) * force

    def inject_grid(self, mass: np.ndarray, scale: float = 1.0):
        """Distribute force proportional to a [0,1] occupancy grid."""
        if mass.shape != self.phi.shape:
            return
        m = mass.astype(np.float32)
        peak = float(m.max())
        if peak < 0.05:
            return
        # sample up to 6 local maxima
        flat = m.copy()
        for _ in range(6):
            j, i = np.unravel_index(int(np.argmax(flat)), flat.shape)
            v = float(flat[j, i])
            if v < 0.12 * peak:
                break
            x = (i + 0.5) / self.size
            y = (j + 0.5) / self.size
            self.inject(x, y, force=scale * v)
            # suppress neighborhood for next peak
            j0, j1 = max(0, j - 2), min(self.size, j + 3)
            i0, i1 = max(0, i - 2), min(self.size, i + 3)
            flat[j0:j1, i0:i1] = 0.0

    def step(self) -> np.ndarray:
        lap = (
            np.roll(self.phi, 1, 0) + np.roll(self.phi, -1, 0) +
            np.roll(self.phi, 1, 1) + np.roll(self.phi, -1, 1)
        ) * 0.25
        self.vel += (lap - self.phi) * self.lmbda
        self.vel *= self.gamma
        self.phi += self.vel
        self._drive *= 0.985
        if self._drive < 0.03:
            self.phi *= 0.992
        np.clip(self.phi, -self.max_abs, self.max_abs, out=self.phi)
        np.clip(self.vel, -self.max_abs, self.max_abs, out=self.vel)
        self.entropy = float(np.std(self.phi))
        return self.phi.copy()


class UltrasonicMapper:
    def __init__(self, base_freq: int = 40000, k: float = 2000.0):
        self.base_freq = base_freq
        self.k = k

    def delta_f(self, field: np.ndarray) -> np.ndarray:
        f = field - float(np.mean(field))
        return f * self.k

    def delta_f_abs(self, field: np.ndarray) -> np.ndarray:
        return np.abs(self.delta_f(field))

    def drive_map(
        self,
        field: np.ndarray,
        csi_spatial: Optional[np.ndarray] = None,
        motion: float = 0.0,
        hold_hz: float = 400.0,
    ) -> np.ndarray:
        base = self.delta_f_abs(field).astype(np.float32)
        if csi_spatial is not None and csi_spatial.shape == field.shape:
            planned = csi_spatial.astype(np.float32) * hold_hz * max(0.15, motion)
            out = np.maximum(base, planned)
        else:
            out = base
        return out

    def region_energies(self, field: np.ndarray, n_emitters: int = 4) -> List[float]:
        h, w = field.shape
        regions = [
            field[0:h // 2, 0:w // 2], field[0:h // 2, w // 2:w],
            field[h // 2:h, 0:w // 2], field[h // 2:h, w // 2:w],
        ]
        return [float(np.mean(np.abs(r))) for r in regions[:n_emitters]]


_REGION_XY = {
    "center": (0.5, 0.5),
    "n": (0.5, 0.75), "s": (0.5, 0.25), "e": (0.75, 0.5), "w": (0.25, 0.5),
    "ne": (0.75, 0.75), "nw": (0.25, 0.75), "se": (0.75, 0.25), "sw": (0.25, 0.25),
    "0": (0.25, 0.25), "1": (0.75, 0.25), "2": (0.25, 0.75), "3": (0.75, 0.75),
}


def _region_xy(name: str, index: int) -> tuple:
    key = str(name).lower().strip()
    if key in _REGION_XY:
        return _REGION_XY[key]
    corners = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    return corners[index % 4]


class EchoGridOS:
    def __init__(
        self,
        size: int = 16,
        body_port: Optional[str] = None,
        csi_port: Optional[int] = None,
        auto_body: bool = False,
    ):
        self.field = EchoFieldOS(size)
        self.mapper = UltrasonicMapper()
        self.t = 0.0
        self.save_path = Path("echo_save.json")
        self.body = None
        self.csi = None
        self.last_obs = 0.0
        self.last_csi_energy = 0.0
        self.csi_packets = 0
        self.last_df_max = 0.0
        self.last_drive_map: Optional[np.ndarray] = None
        self.body_type = None
        self.fuse_agreed = False
        self.fuse_sources = 0
        self.fuse_bands = 0
        self.fuse_conf = 0.0
        self._status_t = 0.0
        self._last_save_bucket = -1
        self.body_connected = False
        self.csi_enabled = False

        if body_port is not None or auto_body:
            try:
                from .body_client import FieldBodyClient
                self.body = FieldBodyClient(port=body_port if body_port else None)
                self.body.connect()
                self.body_connected = True
                print("[EchoGridOS] Field Body attached")
            except Exception as e:
                if body_port is not None:
                    print(f"[EchoGridOS] body unavailable ({e})")
                self.body = None

        if csi_port is not None:
            try:
                from .csi_bridge import CSIBridge
                self.csi = CSIBridge(port=int(csi_port) if csi_port else 4210, grid_size=size)
                self.csi_enabled = self.csi.sock is not None
            except Exception as e:
                print(f"[EchoGridOS] CSI unavailable ({e})")

    def touch(self, x: float, y: float, strength: float = 1.0):
        self.field.inject(x, y, strength)

    def _ingest_body(self, closed_loop: bool):
        if not closed_loop or self.body is None:
            return
        obs = self.body.poll_observation()
        if obs is None:
            return
        self.body_type = obs.get("body_type")
        for i, r in enumerate(obs.get("regions") or []):
            try:
                val = float(r.get("observed", 0.0))
                conf = float(r.get("confidence", 0.5))
            except (TypeError, ValueError):
                continue
            if val < 0.02:
                continue
            x, y = _region_xy(str(r.get("region", i)), i)
            self.field.inject(x, y, force=0.55 * val * max(0.3, conf))
            self.last_obs = max(self.last_obs * 0.9, val)

    def _ingest_csi_belief(self):
        """Primary path: fused occupancy → φ. Tracks secondary."""
        assert self.csi is not None
        self.csi.poll()
        self.last_csi_energy = self.csi.last_energy
        self.csi_packets = self.csi.packet_count
        self.fuse_agreed = bool(getattr(self.csi, "last_fuse_agreed", False))
        self.fuse_sources = int(getattr(self.csi, "last_fuse_sources", 0))
        self.fuse_bands = int(getattr(self.csi, "last_fuse_bands", 0))
        self.fuse_conf = float(self.csi.fuser.fused_confidence()) if hasattr(self.csi, "fuser") else 0.0

        # 1) Belief-field peaks (tomography readout)
        fused = getattr(self.csi.fuser, "last_fused", None)
        if fused is not None and float(fused.occupancy.max()) > 0.08:
            scale = 0.55 * max(0.25, fused.confidence) * (1.25 if fused.agreed else 0.65)
            self.field.inject_grid(fused.occupancy, scale=scale)

        # 2) Confirmed tracks (byproduct, still useful)
        for tr in self.csi.active_tracks():
            if tr.confidence < 0.28:
                continue
            gain = 0.5 if tr.state == "idle" else (0.85 if tr.state == "move" else 1.1)
            if self.fuse_agreed:
                gain *= 1.15
            if tr.energy > 0.04:
                self.field.inject(
                    tr.pos[0], tr.pos[1],
                    force=gain * tr.energy * max(0.3, tr.confidence),
                )

        # 3) Fallback diffuse inject if energy but no structure yet
        if fused is None and self.last_csi_energy > 0.1:
            self.field.inject(0.5, 0.5, force=0.5 * self.last_csi_energy)

    def actuator_map(self, phi: Optional[np.ndarray] = None) -> np.ndarray:
        if phi is None:
            phi = self.field.phi
        spatial = self.csi.spatial_map(self.field.size) if self.csi else None
        dm = self.mapper.drive_map(
            phi, csi_spatial=spatial, motion=self.last_csi_energy, hold_hz=450.0,
        )
        if self.last_drive_map is not None and self.last_drive_map.shape == dm.shape:
            if float(dm.max()) < 15.0 and self.csi_packets > 0:
                dm = np.maximum(dm, self.last_drive_map * 0.92)
            else:
                dm = 0.35 * self.last_drive_map + 0.65 * dm
        self.last_drive_map = dm.astype(np.float32)
        self.last_df_max = float(dm.max())
        return self.last_drive_map

    def step(self, drive_body: bool = False, closed_loop: bool = True) -> np.ndarray:
        if self.csi is not None:
            self._ingest_csi_belief()

        self._ingest_body(closed_loop)
        phi = self.field.step()
        self.actuator_map(phi)

        if closed_loop and self.csi is not None:
            self.csi.closed_loop_feedback(
                entropy=self.field.entropy,
                n_tracks=len(self.csi.active_tracks()),
                motion=self.last_csi_energy,
                df_max=self.last_df_max,
            )

        if drive_body and self.body is not None:
            energies = self.mapper.region_energies(phi, 4)
            best = int(np.argmax(energies))
            if energies[best] > 0.05:
                try:
                    self.body.excite(best)
                except Exception:
                    pass

        self.t += 0.016
        now = time.time()
        if now - self._status_t >= 1.2:
            self._status_t = now
            ntr = len(self.csi.active_tracks()) if self.csi else 0
            print(
                f"[field] entropy={self.field.entropy:.3f}  motion={self.last_csi_energy:.3f}  "
                f"tracks={ntr}  fuse={self.fuse_sources}s/{self.fuse_bands}b "
                f"agreed={self.fuse_agreed}  |Δf|_max={self.last_df_max:.0f}Hz  "
                f"drive={self.field._drive:.2f}  t={self.t:.1f}s"
            )
        return phi

    def save(self, force: bool = False):
        bucket = int(self.t) // 8
        if not force and bucket == self._last_save_bucket:
            return
        self._last_save_bucket = bucket
        with open(self.save_path, "w") as f:
            json.dump({
                "phi": self.field.phi.tolist(),
                "vel": self.field.vel.tolist(),
                "entropy": self.field.entropy,
                "t": self.t,
                "timestamp": time.time(),
            }, f, indent=2)
        print(f"✅ saved → {self.save_path}")

    def close(self):
        if self.body:
            try:
                self.body.passive()
                self.body.close()
            except Exception:
                pass
        if self.csi:
            self.csi.close()
