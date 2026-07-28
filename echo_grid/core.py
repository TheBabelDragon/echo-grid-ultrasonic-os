"""Echo Grid — real inputs only (CSI / body)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, List

import numpy as np


class EchoFieldOS:
    def __init__(self, size: int = 16):
        self.size = size
        self.phi = np.zeros((size, size), dtype=np.float32)
        self.vel = np.zeros((size, size), dtype=np.float32)
        self.lmbda = 0.08
        self.gamma = 0.93
        self.entropy = 0.0
        self.max_abs = 4.0

    def inject(self, x: float, y: float, force: float = 1.0):
        ix = int(np.clip(x * self.size, 0, self.size - 1))
        iy = int(np.clip(y * self.size, 0, self.size - 1))
        force = float(np.clip(force, -2.0, 2.0))
        for j in range(self.size):
            for i in range(self.size):
                dx, dy = i - ix, j - iy
                d2 = dx * dx + dy * dy + 1e-6
                self.vel[j, i] += np.exp(-d2 * 0.15) * force

    def step(self) -> np.ndarray:
        lap = (
            np.roll(self.phi, 1, 0) + np.roll(self.phi, -1, 0) +
            np.roll(self.phi, 1, 1) + np.roll(self.phi, -1, 1)
        ) * 0.25
        self.vel += (lap - self.phi) * self.lmbda
        self.vel *= self.gamma
        self.phi += self.vel
        np.clip(self.phi, -self.max_abs, self.max_abs, out=self.phi)
        np.clip(self.vel, -self.max_abs, self.max_abs, out=self.vel)
        self.entropy = float(np.std(self.phi))
        return self.phi.copy()


class UltrasonicMapper:
    def __init__(self, base_freq: int = 40000, k: float = 2000.0):
        self.base_freq = base_freq
        self.k = k

    def region_energies(self, field: np.ndarray, n_emitters: int = 4) -> List[float]:
        h, w = field.shape
        regions = [
            field[0:h//2, 0:w//2], field[0:h//2, w//2:w],
            field[h//2:h, 0:w//2], field[h//2:h, w//2:w],
        ]
        return [float(np.mean(np.abs(r))) for r in regions[:n_emitters]]


class EchoGridOS:
    def __init__(self, size: int = 16, body_port: Optional[str] = None, csi_port: Optional[int] = None):
        self.field = EchoFieldOS(size)
        self.mapper = UltrasonicMapper()
        self.t = 0.0
        self.save_path = Path("echo_save.json")
        self.body = None
        self.csi = None
        self.last_obs = 0.0
        self.last_csi_energy = 0.0
        self.csi_packets = 0
        self._status_t = 0.0
        self._last_save_bucket = -1
        self.body_connected = False
        self.csi_enabled = False

        if body_port is not None:
            try:
                from .body_client import FieldBodyClient
                self.body = FieldBodyClient(port=body_port if body_port else None)
                self.body.connect()
                self.body_connected = True
                print("[EchoGridOS] body attached")
            except Exception as e:
                print(f"[EchoGridOS] body unavailable ({e})")

        if csi_port is not None:
            try:
                from .csi_bridge import CSIBridge
                self.csi = CSIBridge(port=int(csi_port) if csi_port else 4210)
                self.csi_enabled = self.csi.sock is not None
            except Exception as e:
                print(f"[EchoGridOS] CSI unavailable ({e})")

    def touch(self, x: float, y: float, strength: float = 1.0):
        """Manual / external real inject only — not used by a fake demo loop."""
        self.field.inject(x, y, strength)

    def step(self, drive_body: bool = False, closed_loop: bool = True) -> np.ndarray:
        # Real CSI only
        if self.csi is not None:
            self.csi.poll()
            self.last_csi_energy = self.csi.last_energy
            self.csi_packets = self.csi.packet_count
            for tr in self.csi.active_tracks():
                if tr.energy > 0.05 and tr.confidence > 0.28:
                    x, y = tr.pos
                    self.field.inject(x, y, force=tr.energy * tr.confidence * 0.9)

        # Real ultrasonic body only
        if closed_loop and self.body is not None:
            obs = self.body.poll_observation()
            if obs is not None:
                regions = obs.get("regions", [])
                if regions:
                    val = float(regions[0].get("observed", 0.0))
                    self.last_obs = val
                    if val > 0.01:
                        self.field.inject(0.5, 0.5, force=val * 0.3)

        phi = self.field.step()

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
            modes = []
            if self.body_connected:
                modes.append("body")
            if self.csi_enabled:
                modes.append("csi")
            if not modes:
                modes.append("idle")
            print(
                f"[field] mode={'+'.join(modes)}  entropy={self.field.entropy:.3f}  "
                f"csi={self.last_csi_energy:.3f}  tracks={ntr}  pkts={self.csi_packets}  t={self.t:.1f}s"
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
