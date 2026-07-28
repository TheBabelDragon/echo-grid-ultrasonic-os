#!/usr/bin/env python3
"""
Replay a JSONL capture into Echo Grid via UDP.

  python tools/replay_csi.py
  python tools/replay_csi.py data/session.jsonl --loop --rate 15
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = _ROOT / "data" / "session.jsonl"


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=_DEFAULT,
        help=f"JSONL capture (default: {_DEFAULT})",
    )
    p.add_argument("--port", type=int, default=4210)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--loop", action="store_true")
    a = p.parse_args()

    path = a.file.expanduser()
    if not path.is_absolute():
        path = (_ROOT / path).resolve()

    if not path.is_file():
        raise SystemExit(
            f"Capture file not found:\n  {path}\n\n"
            "Record one first:\n"
            "  python tools/capture_csi.py\n"
            "(ESP CSI nodes must be broadcasting, or you will get 0 packets.)"
        )

    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"No valid rows in {path}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    delay = 1.0 / max(1.0, a.rate)
    print(f"[replay] {len(rows)} rows from {path}")
    print(f"[replay] → {a.host}:{a.port} @ {a.rate} Hz")

    try:
        while True:
            for row in rows:
                pkt = {
                    "node": row.get("node") or "replay",
                    "type": "wifi_csi",
                    "rssi": row.get("rssi", -50),
                    "activity": row.get("activity", 0),
                    "movement_intensity": row.get("movement_intensity", 0),
                    "csi": row.get("csi") or [],
                    "band_movement": row.get("band_movement") or [],
                    "timestamp": int(time.time() * 1000),
                }
                sock.sendto(json.dumps(pkt).encode(), (a.host, a.port))
                time.sleep(delay)
            if not a.loop:
                break
            print("[replay] loop")
    except KeyboardInterrupt:
        pass
    print("done")


if __name__ == "__main__":
    main()
