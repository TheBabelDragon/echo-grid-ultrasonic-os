#!/usr/bin/env python3
"""
Echo Grid — real inputs only

  python main.py --csi
  python main.py --csi --body --drive
"""

import argparse
import time
from echo_grid.core import EchoGridOS


def main():
    p = argparse.ArgumentParser(description="Echo Grid Ultrasonic OS (real-only)")
    p.add_argument("--body", nargs="?", const="", default=None)
    p.add_argument("--csi", nargs="?", const=4210, type=int, default=None)
    p.add_argument("--drive", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--size", type=int, default=16)
    a = p.parse_args()

    if a.csi is None and a.body is None:
        print("No inputs. Use --csi and/or --body (real hardware only).")
        print("  python main.py --csi")
        return

    print("🚀 Echo Grid (real-only)")
    osys = EchoGridOS(size=a.size, body_port=a.body, csi_port=a.csi)
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
