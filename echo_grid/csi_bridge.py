"""CSI Bridge — real packets + arrays for visualization."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .csi_tracks import KalmanTrack, TrackStore


class CSIBridge:
    def __init__(self, port: int = 4210, timeout: float = 0.02):
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.last_packet: Optional[Dict[str, Any]] = None
        self.last_energy: float = 0.0
        self.last_rssi: float = -90.0
        self.packet_count: int = 0
        self.last_rx_time: float = 0.0
        self._logged = 0
        self.tracks = TrackStore(max_tracks=6, ttl_s=2.2)
        # visualization buffers
        self.last_vals: Optional[np.ndarray] = None
        self.last_residual: Optional[np.ndarray] = None
        self.motion_history: List[float] = []
        self._open()

    def _open(self) -> None:
        try:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except OSError:
                pass
            self.sock.bind(("0.0.0.0", self.port))
            self.sock.settimeout(self.timeout)
            print(f"[CSI] listening :{self.port}")
        except OSError as e:
            print(f"[CSI] bind failed: {e}")
            self.sock = None

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def poll(self) -> Optional[Dict[str, Any]]:
        if self.sock is None:
            return None
        latest = None
        addr = None
        while True:
            try:
                data, addr = self.sock.recvfrom(8192)
                try:
                    pkt = json.loads(data.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(pkt, dict):
                    continue
                latest = pkt
                self.tracks.update_from_packet(pkt)
            except socket.timeout:
                break
            except OSError:
                break

        if latest is None:
            if self.last_rx_time and time.time() - self.last_rx_time > 2.0:
                self.last_energy *= 0.88
                if self.last_energy < 0.01:
                    self.last_energy = 0.0
            return None

        self.last_packet = latest
        self.packet_count += 1
        self.last_rx_time = time.time()
        self.last_energy = float(self.tracks.motion_energy)
        self.last_vals = getattr(self.tracks, "last_vals", None)
        self.last_residual = getattr(self.tracks, "last_residual", None)
        self.motion_history.append(self.last_energy)
        if len(self.motion_history) > 200:
            self.motion_history = self.motion_history[-200:]

        try:
            self.last_rssi = float(latest.get("rssi", -90))
        except (TypeError, ValueError):
            self.last_rssi = -90.0

        if self._logged < 6:
            print(f"[CSI] rx #{self.packet_count} from {addr} motion={self.last_energy:.3f}")
            self._logged += 1
        return latest

    def spatial_map(self, size: int = 16) -> np.ndarray:
        """Project CSI residual into a size×size spatial field for display."""
        grid = np.zeros((size, size), dtype=np.float32)
        res = self.last_residual
        vals = self.last_vals
        src = res if res is not None and len(res) > 4 else vals
        if src is None or len(src) < 4:
            # fallback: paint tracks
            for tr in self.active_tracks():
                x, y = tr.pos
                ix = int(np.clip(x * (size - 1), 0, size - 1))
                iy = int(np.clip(y * (size - 1), 0, size - 1))
                for j in range(size):
                    for i in range(size):
                        d2 = (i - ix) ** 2 + (j - iy) ** 2 + 1e-6
                        grid[j, i] += tr.energy * np.exp(-d2 * 0.25)
            return grid

        n = len(src)
        for k, v in enumerate(src):
            # map subcarrier index → x; amplitude → y-band energy
            x = k / max(1, n - 1)
            ix = int(np.clip(x * (size - 1), 0, size - 1))
            amp = float(max(0.0, v))
            # spread vertically proportional to amplitude
            cy = 0.35 + 0.45 * min(1.0, amp)
            iy = int(np.clip(cy * (size - 1), 0, size - 1))
            for j in range(size):
                for i in range(size):
                    d2 = (i - ix) ** 2 + (j - iy) ** 2 * 1.4 + 1e-6
                    grid[j, i] += amp * np.exp(-d2 * 0.35)

        for tr in self.active_tracks():
            x, y = tr.pos
            ix = int(np.clip(x * (size - 1), 0, size - 1))
            iy = int(np.clip(y * (size - 1), 0, size - 1))
            for j in range(size):
                for i in range(size):
                    d2 = (i - ix) ** 2 + (j - iy) ** 2 + 1e-6
                    grid[j, i] += 0.8 * tr.energy * np.exp(-d2 * 0.2)

        m = float(grid.max()) + 1e-9
        return grid / m

    def injection_point(self) -> Tuple[float, float, float]:
        active = self.active_tracks()
        if active:
            x, y = active[0].pos
            return x, y, min(1.2, active[0].energy)
        return 0.5, 0.5, 0.0

    def active_tracks(self) -> List[KalmanTrack]:
        return self.tracks.active()
