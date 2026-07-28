#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS

  python main.py                     # software field only
  python main.py --body              # + closed-loop body
  python main.py --body --drive      # + drive emitters from field regions
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
                        help="Map field regions onto physical emitters")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable observation feedback")
    args = parser.parse_args()

    print("🚀 Echo Grid Ultrasonic OS")
    osys = EchoGridOS(size=16, body_port=args.body)

    try:
        while True:
            t = time.time()
            # moving excitation (can later be replaced by real input)
            x = 0.5 + 0.33 * np.sin(t * 0.62)
            y = 0.5 + 0.33 * np.cos(t * 0.47)
            osys.touch(x, y, strength=0.75)

            osys.step(
                drive_body=args.drive,
                closed_loop=not args.no_loop
            )

            if int(t) % 8 == 0:
                osys.save()

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 shutdown")
        osys.save()
        osys.close()


if __name__ == "__main__":
    main()
