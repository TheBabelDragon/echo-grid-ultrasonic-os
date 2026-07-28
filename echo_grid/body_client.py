"""
Field Body Client

Speaks the shared Field Body Protocol (docs/FIELD_BODY_PROTOCOL.md)
to any compliant physical body (Echo Body, optical-body-s3, …).
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None  # type: ignore


class FieldBodyClient:
    """Minimal host-side client for the Field Body Protocol."""

    def __init__(self, port: Optional[str] = None, baud: int = 115200, timeout: float = 1.0):
        if serial is None:
            raise RuntimeError("pyserial is required: pip install pyserial")

        self.port = port or self._auto_detect_port()
        self.baud = baud
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

    def _auto_detect_port(self) -> str:
        """Best-effort detection of an ESP32 / Arduino-like device."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("cp210", "ch340", "uart", "usb serial", "esp32", "silicon labs")):
                return p.device
        if ports:
            return ports[0].device
        raise RuntimeError("No serial ports found. Plug in the Echo Body and try again.")

    def connect(self) -> None:
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(1.2)  # allow ESP32 reset + banner
        self._ser.reset_input_buffer()
        print(f"[BodyClient] connected to {self.port}")

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _send(self, line: str) -> None:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Not connected")
        self._ser.write((line.strip() + "\n").encode("utf-8"))
        self._ser.flush()

    def excite(self, emitter_id: int) -> None:
        self._send(f"EXCITE {emitter_id}")

    def map(self) -> None:
        self._send("MAP")

    def verify(self) -> None:
        self._send("VERIFY")

    def passive(self) -> None:
        self._send("PASSIVE")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
