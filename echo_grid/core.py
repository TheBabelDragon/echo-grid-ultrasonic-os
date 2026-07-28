"""
Echo Grid Ultrasonic OS — Core Kernel
Field evolution + ultrasonic mapping + optional physical body control
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


class EchoFieldOS:
    """2D coupled oscillator lattice — the computational brain."""

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
                influence = np.exp(-d2 * 0.15)
                self.vel[j, i] += influence * force

    def step(self) -> np.ndarray:
        lap = (
            np.roll(self.phi, 1, 0)
            + np.roll(self.phi, -1, 0)
            + np.roll(self.phi, 1, 1)
            + np.roll(self.phi, -1, 1)
        ) * 0.25

        self.vel += (lap - self.phi) * self.lmbda
        self.vel *= self.gamma
        self.phi += self.vel
        self.entropy = float(np.std(self.phi))
        return self.phi.copy()


class UltrasonicMapper:
    """Maps field state → ultrasonic parameters."""

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
                    "amp": min(1.0, abs(v)),
                    "phase": v,
                })
        return packets


class EchoGridOS:
    """Main controller / OS kernel."""

    def __init__(self, size: int = 16, body_port: Optional[str] = None):
        self.field = EchoFieldOS(size)
        self.mapper = UltrasonicMapper()
        self.t = 0.0
        self.save_path = Path("echo_save.json")
        self.body = None

        if body_port is not None or body_port == "":  # explicit request to attach body
            try:
                from .body_client import FieldBodyClient
                self.body = FieldBodyClient(port=body_port if body_port else None)
                self.body.connect()
                print("[EchoGridOS] physical body attached")
            except Exception as e:
                print(f"[EchoGridOS] body not available: {e}")
                self.body = None

    def touch(self, x: float, y: float, strength: float = 1.0):
        self.field.inject(x, y, strength)

    def step(self, drive_body: bool = False) -> np.ndarray:
        phi = self.field.step()
        packets = self.mapper.encode(phi)

        # Optional: map strongest region onto a physical emitter
        if drive_body and self.body is not None:
            flat = phi.ravel()
            strongest = int(np.argmax(np.abs(flat)))
            # crude mapping: use a few emitters
            emitter_id = strongest % 4
            try:
                self.body.excite(emitter_id)
            except Exception:
                pass

        self.t += 0.016
        return phi

    def save(self):
        data = {
            "phi": self.field.phi.tolist(),
            "vel": self.field.vel.tolist(),
            "lmbda": self.field.lmbda,
            "gamma": self.field.gamma,
            "timestamp": time.time(),
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ State saved → {self.save_path}")

    def load(self):
        if not self.save_path.exists():
            print("⚠️  No savepoint found")
            return
        with open(self.save_path, "r") as f:
            data = json.load(f)
        self.field.phi = np.array(data["phi"], dtype=np.float32)
        self.field.vel = np.array(data["vel"], dtype=np.float32)
        self.field.lmbda = data.get("lmbda", 0.08)
        self.field.gamma = data.get("gamma", 0.92)
        print(f"✅ State loaded from {self.save_path}")

    def close(self):
        if self.body:
            try:
                self.body.passive()
                self.body.close()
            except Exception:
                pass
