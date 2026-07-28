#!/usr/bin/env python3
"""
Echo Grid — CSI on by default

  python main.py
  python main.py --body --drive
  python main.py --no-csi
"""

import argparse
import time
from echo_grid.core import EchoGridOS


def main():
    p = argparse.ArgumentParser(description="Echo Grid Ultrasonic OS")
    p.add_argument("--body", nargs="?", const="", default=None)
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=4210)
    p.add_argument("--no-csi", action="store_true")
    p.add_argument("--drive", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()

    csi_port = None if a.no_csi else a.csi
    print("🚀 Echo Grid")
    print(f"   CSI: {'off' if csi_port is None else f'UDP :{csi_port}'}")
    print(f"   body: {'on' if a.body is not None else 'off'}")

    osys = EchoGridOS(size=a.size, body_port=a.body, csi_port=csi_port)
    try:
        while True:
            osys.step(drive_body=a.drive, closed_loop=not a.no_loop)
            osys.save()
            time.sleep(0.016)
    except KeyboardInterrupt:
        print("\nshutdown")
        osys.save(force=True)
        osys.close()


if __name__ == "__main__":
    main()
