#!/usr/bin/env python3
"""
Capture CSI packets to JSONL with optional labels.

  python tools/capture_csi.py
  python tools/capture_csi.py -o data/session.jsonl

Labels (type + Enter):  e=empty  1=one  2=two  w=walk  s=still  q=quit
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time
from pathlib import Path

# repo root = parent of tools/
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _ROOT / "data" / "session.jsonl"

LABEL_KEYS = {
    "e": "empty",
    "1": "one_person",
    "2": "two_person",
    "3": "three_person",
    "w": "walk",
    "s": "still",
    "u": "unknown",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o", "--output",
        default=str(_DEFAULT_OUT),
        help=f"JSONL path (default: {_DEFAULT_OUT})",
    )
    p.add_argument("--port", type=int, default=4210)
    a = p.parse_args()

    out = Path(a.output).expanduser()
    if not out.is_absolute():
        out = (_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", a.port))
    except OSError as e:
        raise SystemExit(
            f"Cannot bind UDP :{a.port} ({e}).\n"
            "Stop the dashboard if it is holding the port, or pass --port 4212."
        ) from e
    sock.settimeout(0.3)

    label = "unknown"
    n = 0
    print(f"[capture] listening :{a.port}")
    print(f"[capture] writing → {out}")
    print("labels: e empty | 1/2/3 persons | w walk | s still | q quit")
    print(f"current label={label}")

    try:
        with out.open("a", encoding="utf-8") as f:
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line == "q":
                        break
                    if line in LABEL_KEYS:
                        label = LABEL_KEYS[line]
                        print(f"[label] {label}")
                    elif line:
                        label = line
                        print(f"[label] {label}")

                try:
                    data, addr = sock.recvfrom(8192)
                except socket.timeout:
                    continue

                try:
                    pkt = json.loads(data.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(pkt, dict):
                    continue

                row = {
                    "t": time.time(),
                    "label": label,
                    "src": addr[0],
                    "node": pkt.get("node"),
                    "rssi": pkt.get("rssi"),
                    "activity": pkt.get("activity"),
                    "movement_intensity": pkt.get("movement_intensity"),
                    "csi": pkt.get("csi"),
                    "band_movement": pkt.get("band_movement"),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    print(f"[capture] {n} packets  label={label}  node={row.get('node')}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    print(f"done — {n} rows → {out}")


if __name__ == "__main__":
    main()
