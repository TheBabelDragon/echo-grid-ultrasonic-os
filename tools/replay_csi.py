#!/usr/bin/env python3
"""Replay a JSONL capture into Echo Grid (UDP localhost or in-process)."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open() as f:
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
    p.add_argument("file", type=Path)
    p.add_argument("--port", type=int, default=4210)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--rate", type=float, default=20.0, help="packets per second")
    p.add_argument("--loop", action="store_true")
    a = p.parse_args()

    rows = load_rows(a.file)
    if not rows:
        raise SystemExit("no rows")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    delay = 1.0 / max(1.0, a.rate)
    print(f"[replay] {len(rows)} rows → {a.host}:{a.port} @ {a.rate} Hz")

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
