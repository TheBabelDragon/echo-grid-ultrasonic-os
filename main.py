#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS — production entry point

  python main.py                     # software field only
  python main.py --body              # + closed-loop body
  python main.py --body --drive      # + drive emitters from field
"""

import argparse
import time
import numpy as np
from echo_grid.core import EchoGridOS


def main():
    parser = argparse.ArgumentParser(description="Echo Grid Ultrasonic OS")
    parser.add_argument("--body", nargs="?", const="", default=None,
                        help="Attach Echo Body (optional serial port)")
    parser.add_argument("--drive", action="store_true",
                        help="Map field regions onto physical emitters")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable observation feedback")
    parser.add_argument("--size", type=int, default=16,
                        help="Field lattice size (default 16)")
    args = parser.parse_args()

    print("🚀 Echo Grid Ultrasonic OS")
    print(f"   field={args.size}x{args.size}  body={'yes' if args.body is not None else 'no'}  drive={args.drive}")

    osys = EchoGridOS(size=args.size, body_port=args.body)

    try:
        while True:
            t = time.time()

            # Moving external excitation (demo drive)
            x = 0.5 + 0.30 * np.sin(t * 0.55)
            y = 0.5 + 0.30 * np.cos(t * 0.41)
            osys.touch(x, y, strength=0.55)

            osys.step(
                drive_body=args.drive,
                closed_loop=not args.no_loop
            )

            osys.save()   # internal rate-limit prevents spam

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 shutdown")
        osys.save(force=True)
        osys.close()


if __name__ == "__main__":
    main()
