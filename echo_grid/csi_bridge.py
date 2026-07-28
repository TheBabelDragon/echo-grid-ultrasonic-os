"""
CSI Bridge — UDP :4210 from wifi-sensing-system ESP32 nodes.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional, Tuple


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
        self._baseline: Optional[List[float]] = None
        self._logged = 0
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
            print(f"[CSI] listening UDP 0.0.0.0:{self.port} (broadcast OK)")
        except OSError as e:
            print(f"[CSI] bind failed on :{self.port} ({e})")
            print("      Is another process using 4210?  sudo ss -ulnp | grep 4210")
            self.sock = None

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _energy_from_packet(self, pkt: Dict[str, Any]) -> float:
        # Prefer node-computed features (stable)
        candidates = []
        for key in ("movement_intensity", "activity", "confidence"):
            if key in pkt and pkt[key] is not None:
                try:
                    candidates.append(float(pkt[key]))
                except (TypeError, ValueError):
                    pass
        if candidates:
            return float(min(1.0, max(0.0, max(candidates))))

        csi = pkt.get("csi") or []
        try:
            csi_f = [float(x) for x in csi]
        except (TypeError, ValueError):
            return 0.0

        if not csi_f:
            return 0.0

        if self._baseline is None or len(self._baseline) != len(csi_f):
            self._baseline = list(csi_f)
            return 0.05  # first frame — small presence pulse

        dev = [abs(a - b) for a, b in zip(csi_f, self._baseline)]
        energy = sum(dev) / max(1, len(dev))
        alpha = 0.05
        self._baseline = [
            (1 - alpha) * b + alpha * a for a, b in zip(csi_f, self._baseline)
        ]
        return float(min(1.0, energy * 4.0))

    def poll(self) -> Optional[Dict[str, Any]]:
        if self.sock is None:
            return None

        latest = None
        latest_addr = None
        while True:
            try:
                data, addr = self.sock.recvfrom(8192)
                try:
                    pkt = json.loads(data.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    if self._logged < 3:
                        print(f"[CSI] bad JSON from {addr}: {data[:80]!r}")
                        self._logged += 1
                    continue
                if not isinstance(pkt, dict):
                    continue
                latest = pkt
                latest_addr = addr
            except socket.timeout:
                break
            except OSError:
                break

        if latest is None:
            # soft decay if silent > 2s
            if self.last_rx_time and (time.time() - self.last_rx_time) > 2.0:
                self.last_energy *= 0.92
                if self.last_energy < 0.01:
                    self.last_energy = 0.0
            return None

        self.last_packet = latest
        self.packet_count += 1
        self.last_rx_time = time.time()
        self.last_energy = self._energy_from_packet(latest)

        try:
            self.last_rssi = float(latest.get("rssi", -90))
        except (TypeError, ValueError):
            self.last_rssi = -90.0

        if self._logged < 5:
            print(
                f"[CSI] rx #{self.packet_count} from {latest_addr}  "
                f"energy={self.last_energy:.3f}  rssi={self.last_rssi:.0f}  "
                f"keys={list(latest.keys())[:8]}"
            )
            self._logged += 1

        return latest

    def injection_point(self) -> Tuple[float, float, float]:
        force = self.last_energy
        if force < 0.015:
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
            right = sum(csi_f[mid:]) / max(1, len(csi_f) - mid)
            total = left + right + 1e-6
            x = 0.30 + 0.40 * (right / total)
            y = 0.40 + 0.20 * force
        else:
            x, y = 0.5, 0.5

        return (x, y, min(1.2, force))
