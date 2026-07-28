"""
Field Body Client — host side of the Field Body Protocol
Production-hardened: accepts compact OBS and rich optical JSON.
"""

from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None  # type: ignore


class FieldBodyClient:
    def __init__(self, port: Optional[str] = None, baud: int = 115200, timeout: float = 0.25):
        if serial is None:
            raise RuntimeError("pyserial is required: pip install pyserial")
        self.port = port or self._auto_detect_port()
        self.baud = baud
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self.last_observation: Optional[Dict[str, Any]] = None

    def _auto_detect_port(self) -> str:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("cp210", "ch340", "uart", "usb serial", "esp32", "silicon labs")):
                return p.device
        if ports:
            return ports[0].device
        raise RuntimeError("No serial ports found")

    def connect(self) -> None:
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(1.0)
        self._ser.reset_input_buffer()
        print(f"[BodyClient] connected → {self.port}")

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass

    def _send(self, line: str) -> None:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Not connected")
        self._ser.write((line.strip() + "\n").encode("utf-8"))
        self._ser.flush()

    def excite(self, emitter_id: int) -> None:
        self._send(f"EXCITE {int(emitter_id)}")

    def map(self) -> None:
        self._send("MAP")

    def verify(self) -> None:
        self._send("VERIFY")

    def passive(self) -> None:
        self._send("PASSIVE")

    def _normalize_obs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "regions" in data:
            return data

        regions = data.get("field_regions") or []
        best = None
        best_val = -1.0
        for r in regions:
            try:
                val = float(r.get("observed", 0.0))
            except (TypeError, ValueError):
                val = 0.0
            if val > best_val:
                best_val = val
                best = r

        return {
            "body_id": data.get("body_id", "unknown"),
            "body_type": data.get("body_type", "optical"),
            "excitation_id": data.get("excitation_id", -1),
            "geometry_state": data.get("geometry_state", "unknown"),
            "health": data.get("health", "ok"),
            "regions": [best] if best else [{"region": "none", "observed": 0.0, "confidence": 0.0}],
        }

    def poll_observation(self) -> Optional[Dict[str, Any]]:
        if not self._ser or not self._ser.is_open:
            return None

        while self._ser.in_waiting:
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                payload = None
                if raw.startswith("OBS "):
                    payload = raw[4:]
                elif raw.startswith("{") and ("body_type" in raw or "field_regions" in raw or "regions" in raw):
                    payload = raw

                if payload is None:
                    continue

                data = json.loads(payload)
                obs = self._normalize_obs(data)
                self.last_observation = obs
                return obs
            except Exception:
                continue
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
