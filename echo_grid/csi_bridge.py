"""CSI Bridge — :4210 in, closed-loop cmds + field telemetry on :4211."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .csi_tracks import KalmanTrack, TrackStore


class CSIBridge:
    def __init__(self, port: int = 4210, cmd_port: int = 4211, timeout: float = 0.02):
        self.port = port
        self.cmd_port = cmd_port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.cmd_sock: Optional[socket.socket] = None
        self.last_packet: Optional[Dict[str, Any]] = None
        self.last_energy: float = 0.0
        self.last_rssi: float = -90.0
        self.packet_count: int = 0
        self.last_rx_time: float = 0.0
        self.last_addr: Optional[Tuple[str, int]] = None
        self._logged = 0
        self._last_cmd_t = 0.0
        self._last_mode = ""
        self.tracks = TrackStore(max_tracks=6, ttl_s=2.2)
        self.last_vals: Optional[np.ndarray] = None
        self.last_residual: Optional[np.ndarray] = None
        self.motion_history: List[float] = []
        self.cmd_count = 0
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

        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            print(f"[CSI] command+telemetry → :{self.cmd_port}")
        except OSError as e:
            print(f"[CSI] cmd socket failed: {e}")
            self.cmd_sock = None

    def close(self) -> None:
        for s in (self.sock, self.cmd_sock):
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self.sock = None
        self.cmd_sock = None

    def send_command(self, cmd: str, **fields: Any) -> bool:
        if self.cmd_sock is None:
            return False
        payload = {"type": "echo_cmd", "cmd": cmd, **fields}
        data = json.dumps(payload).encode("utf-8")
        targets = []
        if self.last_addr:
            targets.append((self.last_addr[0], self.cmd_port))
        targets.append(("255.255.255.255", self.cmd_port))
        ok = False
        for host, port in targets:
            try:
                self.cmd_sock.sendto(data, (host, port))
                ok = True
            except OSError:
                continue
        if ok:
            self.cmd_count += 1
        return ok

    def closed_loop_feedback(
        self,
        entropy: float,
        n_tracks: int,
        motion: float,
        df_max: float = 0.0,
    ) -> None:
        now = time.time()
        if now - self._last_cmd_t < 0.7:
            return
        self._last_cmd_t = now

        # Always push field telemetry so CYD can mirror the grid
        self.send_command(
            "field",
            entropy=round(float(entropy), 3),
            tracks=int(n_tracks),
            motion=round(float(motion), 3),
            df_max=round(float(df_max), 1),
        )

        if n_tracks >= 2 or motion > 0.55 or entropy > 0.45:
            mode = "boost"
            level = float(min(1.0, 0.4 + 0.4 * motion + 0.2 * min(n_tracks, 3) / 3))
            self.send_command("boost", level=round(level, 2))
        elif motion < 0.08 and entropy < 0.12 and n_tracks == 0:
            mode = "quiet"
            self.send_command("quiet")
        else:
            mode = "nominal"
            interval = int(max(150, min(900, 700 - 500 * motion)))
            self.send_command("set_rate", interval_ms=interval)

        if mode != self._last_mode:
            print(f"[CSI] closed-loop → {mode} (H={entropy:.2f} tracks={n_tracks} df={df_max:.0f})")
            self._last_mode = mode

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
                if addr:
                    self.last_addr = addr
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
        grid = np.zeros((size, size), dtype=np.float32)
        res = self.last_residual
        vals = self.last_vals
        src = res if res is not None and len(res) > 4 else vals
        if src is None or len(src) < 4:
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
            x = k / max(1, n - 1)
            ix = int(np.clip(x * (size - 1), 0, size - 1))
            amp = float(max(0.0, v))
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
