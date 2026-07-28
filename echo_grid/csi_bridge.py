"""
CSI Bridge — ingest WiFi CSI packets from wifi-sensing-system nodes.

Compatible with the UDP JSON contract used by:
  wifi-sensing-system/esp32/esp32_csi_udp_sender.ino
  wifi-sensing-system/ingestion/ingestor.py

Default port: 4210
Expected packet shape:
  {"node": "...", "rssi": -55, "csi": [0.1, ...], "type": "wifi_csi", ...}
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple


class CSIBridge:
    def __init__(self, port: int = 4210, timeout: float = 0.01):
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.last_packet: Optional[Dict[str, Any]] = None
        self.last_energy: float = 0.0
        self.last_rssi: float = -90.0
        self.packet_count: int = 0
        self._baseline: Optional[List[float]] = None
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
            self.sock.bind(("0.0.0.0", self.port))
            self.sock.settimeout(self.timeout)
            print(f"[CSI] listening UDP :{self.port}")
        except OSError as e:
            print(f"[CSI] bind failed on :{self.port} ({e}) — CSI input disabled")
            self.sock = None

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def poll(self) -> Optional[Dict[str, Any]]:
        """Non-blocking read of the latest CSI packet."""
        if self.sock is None:
            return None

        latest = None
        # Drain queue, keep newest
        while True:
            try:
                data, _addr = self.sock.recvfrom(4096)
                pkt = json.loads(data.decode("utf-8", errors="ignore"))
                latest = pkt
            except socket.timeout:
                break
            except Exception:
                break

        if latest is None:
            return None

        self.last_packet = latest
        self.packet_count += 1

        csi = latest.get("csi") or []
        try:
            csi_f = [float(x) for x in csi]
        except (TypeError, ValueError):
            csi_f = []

        if csi_f:
            if self._baseline is None or len(self._baseline) != len(csi_f):
                self._baseline = list(csi_f)
            # Motion energy = mean absolute deviation from slow baseline
            dev = [abs(a - b) for a, b in zip(csi_f, self._baseline)]
            energy = sum(dev) / max(1, len(dev))
            # Slow baseline adapt
            alpha = 0.02
            self._baseline = [
                (1 - alpha) * b + alpha * a for a, b in zip(csi_f, self._baseline)
            ]
            self.last_energy = float(min(1.0, energy * 3.0))
        else:
            self.last_energy = 0.0

        try:
            self.last_rssi = float(latest.get("rssi", -90))
        except (TypeError, ValueError):
            self.last_rssi = -90.0

        return latest

    def injection_point(self) -> Tuple[float, float, float]:
        """
        Map CSI → (x, y, force) for field injection.
        Uses subcarrier energy distribution as a crude spatial bias when possible.
        """
        force = self.last_energy
        if force < 0.02:
            return (0.5, 0.5, 0.0)

        pkt = self.last_packet or {}
        csi = pkt.get("csi") or []
        try:
            csi_f = [float(x) for x in csi]
        except (TypeError, ValueError):
            csi_f = []

        if len(csi_f) >= 8:
            mid = len(csi_f) // 2
            left = sum(csi_f[:mid]) / mid
            right = sum(csi_f[mid:]) / (len(csi_f) - mid)
            total = left + right + 1e-6
            x = 0.35 + 0.30 * (right / total)   # bias toward stronger side
            y = 0.45 + 0.10 * (self.last_energy)
        else:
            x, y = 0.5, 0.5

        # RSSI modulates force slightly (closer / stronger link → more force)
        rssi_boost = max(0.0, min(1.0, (self.last_rssi + 90.0) / 50.0))
        force = min(1.2, force * (0.7 + 0.5 * rssi_boost))
        return (x, y, force)
