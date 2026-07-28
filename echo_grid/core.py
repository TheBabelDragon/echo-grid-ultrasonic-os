"""
Echo Grid Ultrasonic OS — Core Kernel (closed-loop + spatial mapping)
"""

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
        self.gamma = 0.92
        self.entropy = 0.0

    def inject(self, x: float, y: float, force: float = 1.0):
        ix = int(np.clip(x * self.size, 0, self.size - 1))
        iy = int(np.clip(y * self.size, 0, self.size - 1))
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
                    "x": x, "y": y,
                    "freq": self.base_freq + v * self.k,
                    "amp": min(1.0, abs(v)),
                    "phase": v,
                })
        return packets

    def region_energies(self, field: np.ndarray, n_emitters: int = 4) -> List[float]:
        """Collapse the 2D field into n_emitters spatial bins (simple quadrant-style)."""
        h, w = field.shape
        energies = []
        # For 4 emitters we use 2×2 spatial regions
        if n_emitters == 4:
            regions = [
                field[0:h//2, 0:w//2],
                field[0:h//2, w//2:w],
                field[h//2:h, 0:w//2],
                field[h//2:h, w//2:w],
            ]
            for r in regions:
                energies.append(float(np.mean(np.abs(r))))
        else:
            # fallback: uniform slices
            flat = np.abs(field).ravel()
            chunk = max(1, len(flat) // n_emitters)
            for i in range(n_emitters):
                energies.append(float(np.mean(flat[i*chunk:(i+1)*chunk])))
        return energies


class EchoGridOS:
    def __init__(self, size: int = 16, body_port: Optional[str] = None):
        self.field = EchoFieldOS(size)
        self.mapper = UltrasonicMapper()
        self.t = 0.0
        self.save_path = Path("echo_save.json")
        self.body = None
        self.last_obs = 0.0
        self._status_t = 0.0

        if body_port is not None:
            try:
                from .body_client import FieldBodyClient
                self.body = FieldBodyClient(port=body_port if body_port else None)
                self.body.connect()
                print("[EchoGridOS] body attached — closed loop active")
            except Exception as e:
                print(f"[EchoGridOS] body unavailable: {e}")
                self.body = None

    def touch(self, x: float, y: float, strength: float = 1.0):
        self.field.inject(x, y, strength)

    def step(self, drive_body: bool = False, closed_loop: bool = True) -> np.ndarray:
        # --- closed-loop observation intake ---
        if closed_loop and self.body is not None:
            obs = self.body.poll_observation()
            if obs is not None:
                regions = obs.get("regions", [])
                if regions:
                    val = float(regions[0].get("observed", 0.0))
                    self.last_obs = val
                    # gentle, stable feedback into the centre
                    self.field.inject(0.5, 0.5, force=val * 0.35)

        # --- evolve pure field ---
        phi = self.field.step()

        # --- drive physical emitters from spatial regions ---
        if drive_body and self.body is not None:
            energies = self.mapper.region_energies(phi, n_emitters=4)
            # fire the currently strongest region
            best = int(np.argmax(energies))
            if energies[best] > 0.04:          # small dead-zone
                try:
                    self.body.excite(best)
                except Exception:
                    pass

        self.t += 0.016

        # --- occasional live status ---
        if time.time() - self._status_t > 1.5:
            self._status_t = time.time()
            print(
                f"[field] entropy={self.field.entropy:.3f}  "
                f"obs={self.last_obs:.3f}  "
                f"t={self.t:.1f}s"
            )

        return phi

    def save(self):
        data = {
            "phi": self.field.phi.tolist(),
            "vel": self.field.vel.tolist(),
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
