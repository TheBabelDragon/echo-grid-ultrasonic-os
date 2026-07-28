#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS — Entry Point

Usage:
  python main.py                  # pure software field
  python main.py --body           # try to attach Echo Body over serial
  python main.py --body /dev/ttyUSB0
"""

import argparse
import time
import numpy as np
from echo_grid.core import EchoGridOS


def main():
    parser = argparse.ArgumentParser(description="Echo Grid Ultrasonic OS")
    parser.add_argument("--body", nargs="?", const="", default=None,
                        help="Attach physical Echo Body (optional port)")
    parser.add_argument("--drive", action="store_true",
                        help="Map live field onto physical emitters")
    args = parser.parse_args()

    print("🚀 Echo Grid Ultrasonic OS v1.1")
    osys = EchoGridOS(size=16, body_port=args.body)

    try:
        while True:
            t = time.time()
            x = 0.5 + 0.35 * np.sin(t * 0.7)
            y = 0.5 + 0.35 * np.cos(t * 0.5)
            osys.touch(x, y, strength=0.85)

            field = osys.step(drive_body=args.drive)

            if int(t) % 5 == 0:
                osys.save()

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 Echo Grid OS shutting down.")
        osys.save()
        osys.close()


if __name__ == "__main__":
    main()
