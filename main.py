#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS

  python main.py                     # pure software
  python main.py --body              # attach body + closed loop
  python main.py --body --drive      # also map field onto emitters
"""

import argparse
import time
import numpy as np
from echo_grid.core import EchoGridOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", nargs="?", const="", default=None,
                        help="Attach Echo Body (optional serial port)")
    parser.add_argument("--drive", action="store_true",
                        help="Drive physical emitters from the field")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable closed-loop feedback")
    args = parser.parse_args()

    print("🚀 Echo Grid Ultrasonic OS — closed-loop edition")
    osys = EchoGridOS(size=16, body_port=args.body)

    try:
        while True:
            t = time.time()
            # synthetic external drive (can be replaced by real input later)
            x = 0.5 + 0.32 * np.sin(t * 0.65)
            y = 0.5 + 0.32 * np.cos(t * 0.48)
            osys.touch(x, y, strength=0.7)

            field = osys.step(
                drive_body=args.drive,
                closed_loop=not args.no_loop
            )

            if int(t) % 6 == 0:
                osys.save()

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 shutdown")
        osys.save()
        osys.close()


if __name__ == "__main__":
    main()
