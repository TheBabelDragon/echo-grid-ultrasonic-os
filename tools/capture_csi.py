#!/usr/bin/env python3
"""
Capture CSI packets to JSONL with optional labels.

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
    p.add_argument("-o", "--output", default="data/csi_capture.jsonl")
    p.add_argument("--port", type=int, default=4210)
    a = p.parse_args()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(0.3)

    label = "unknown"
    n = 0
    print(f"[capture] :{a.port} → {out}")
    print("labels: e empty | 1/2/3 persons | w walk | s still | q quit")
    print(f"current label={label}")

    with out.open("a") as f:
        try:
            while True:
                # non-blocking stdin label
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

    sock.close()
    print(f"done — {n} rows → {out}")


if __name__ == "__main__":
    main()
