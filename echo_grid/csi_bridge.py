"""CSI Bridge — real packets only."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

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
            print(f"[CSI] listening :{self.port} (real-only tracker)")
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
        try:
            self.last_rssi = float(latest.get("rssi", -90))
        except (TypeError, ValueError):
            self.last_rssi = -90.0

        if self._logged < 8:
            print(
                f"[CSI] rx #{self.packet_count} from {addr}  "
                f"motion={self.last_energy:.3f}  tracks={len(self.tracks.active())}"
            )
            self._logged += 1
        return latest

    def injection_point(self) -> Tuple[float, float, float]:
        active = self.active_tracks()
        if active:
            x, y = active[0].pos
            return x, y, min(1.2, active[0].energy)
        return 0.5, 0.5, 0.0

    def active_tracks(self) -> List[KalmanTrack]:
        return self.tracks.active()
