"""
Echo Grid Ultrasonic OS — Core Kernel (production-hardened)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, List

import numpy as np


class EchoFieldOS:
    """2D coupled oscillator lattice with soft energy limits."""

    def __init__(self, size: int = 16):
        self.size = size
        self.phi = np.zeros((size, size), dtype=np.float32)
        self.vel = np.zeros((size, size), dtype=np.float32)
        self.lmbda = 0.08
        self.gamma = 0.94          # slightly stronger damping
        self.entropy = 0.0
        self.max_abs = 4.0         # soft clamp on field amplitude

    def inject(self, x: float, y: float, force: float = 1.0):
        ix = int(np.clip(x * self.size, 0, self.size - 1))
        iy = int(np.clip(y * self.size, 0, self.size - 1))
        force = float(np.clip(force, -2.0, 2.0))

        for j in range(self.size):
            for i in range(self.size):
                dx = i - ix
                dy = j - iy
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

        # Soft energy clamp — prevents runaway in long runs
        np.clip(self.phi, -self.max_abs, self.max_abs, out=self.phi)
        np.clip(self.vel, -self.max_abs, self.max_abs, out=self.vel)

        self.entropy = float(np.std(self.phi))
        return self.phi.copy()


class UltrasonicMapper:
    def __init__(self, base_freq: int = 40000, k: float = 2000.0):
        self.base_freq = base_freq
        self.k = k

    def encode(self, field: np.ndarray) -> list[dict]:
        packets = []
        for y in range(field.shape[0]):
            for x in range(field.shape[1]):
                v = float(field[y, x])
                packets.append({
                    "x": x,
                    "y": y,
                    "freq": self.base_freq + v * self.k,
                    "amp": min(1.0, abs(v) / 4.0),
                    "phase": v,
                })
        return packets

    def region_energies(self, field: np.ndarray, n_emitters: int = 4) -> List[float]:
        h, w = field.shape
        if n_emitters == 4:
            regions = [
                field[0:h//2, 0:w//2],
                field[0:h//2, w//2:w],
                field[h//2:h, 0:w//2],
                field[h//2:h, w//2:w],
            ]
            return [float(np.mean(np.abs(r))) for r in regions]

        flat = np.abs(field).ravel()
        chunk = max(1, len(flat) // n_emitters)
        return [float(np.mean(flat[i*chunk:(i+1)*chunk])) for i in range(n_emitters)]


class EchoGridOS:
    def __init__(self, size: int = 16, body_port: Optional[str] = None):
        self.field = EchoFieldOS(size)
        self.mapper = UltrasonicMapper()
        self.t = 0.0
        self.save_path = Path("echo_save.json")
        self.body = None
        self.last_obs = 0.0
        self._status_t = 0.0
        self._last_save_bucket = -1
        self.body_connected = False

        if body_port is not None:
            try:
                from .body_client import FieldBodyClient
                self.body = FieldBodyClient(port=body_port if body_port else None)
                self.body.connect()
                self.body_connected = True
                print("[EchoGridOS] body attached — closed loop active")
            except Exception as e:
                print(f"[EchoGridOS] body unavailable ({e}) — software-only mode")
                self.body = None
                self.body_connected = False

    def touch(self, x: float, y: float, strength: float = 1.0):
        self.field.inject(x, y, strength)

    def step(self, drive_body: bool = False, closed_loop: bool = True) -> np.ndarray:
        # Closed-loop intake
        if closed_loop and self.body is not None:
            obs = self.body.poll_observation()
            if obs is not None:
                regions = obs.get("regions", [])
                if regions:
                    val = float(regions[0].get("observed", 0.0))
                    self.last_obs = val
                    if val > 0.01:  # ignore pure noise floor
                        self.field.inject(0.5, 0.5, force=val * 0.30)

        phi = self.field.step()

        # Drive emitters from spatial regions
        if drive_body and self.body is not None:
            energies = self.mapper.region_energies(phi, n_emitters=4)
            best = int(np.argmax(energies))
            if energies[best] > 0.05:
                try:
                    self.body.excite(best)
                except Exception:
                    pass

        self.t += 0.016

        # Status line (~1.5 s)
        now = time.time()
        if now - self._status_t >= 1.5:
            self._status_t = now
            body_state = "body" if self.body_connected else "soft"
            print(
                f"[field] mode={body_state}  "
                f"entropy={self.field.entropy:.3f}  "
                f"obs={self.last_obs:.3f}  "
                f"t={self.t:.1f}s"
            )

        return phi

    def save(self, force: bool = False):
        """Save at most once per 8-second bucket unless forced."""
        bucket = int(self.t) // 8
        if not force and bucket == self._last_save_bucket:
            return
        self._last_save_bucket = bucket

        data = {
            "phi": self.field.phi.tolist(),
            "vel": self.field.vel.tolist(),
            "entropy": self.field.entropy,
            "t": self.t,
            "timestamp": time.time(),
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ saved → {self.save_path}")

    def close(self):
        if self.body:
            try:
                self.body.passive()
                self.body.close()
            except Exception:
                pass
            self.body_connected = False
