#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS

  python main.py
  python main.py --csi
  python main.py --body --csi --drive
"""

import argparse
import time
import numpy as np
from echo_grid.core import EchoGridOS


def main():
    parser = argparse.ArgumentParser(description="Echo Grid Ultrasonic OS")
    parser.add_argument("--body", nargs="?", const="", default=None,
                        help="Attach Echo Body (optional serial port)")
    parser.add_argument("--csi", nargs="?", const=4210, type=int, default=None,
                        help="Enable CSI input (UDP port, default 4210)")
    parser.add_argument("--drive", action="store_true",
                        help="Map field regions onto physical emitters")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable ultrasonic observation feedback")
    parser.add_argument("--no-demo", action="store_true",
                        help="Disable synthetic moving excitation (CSI/body only)")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()

    print("🚀 Echo Grid Ultrasonic OS")
    print(
        f"   field={args.size}x{args.size}  "
        f"body={'yes' if args.body is not None else 'no'}  "
        f"csi={'yes:'+str(args.csi) if args.csi is not None else 'no'}  "
        f"drive={args.drive}"
    )

    osys = EchoGridOS(size=args.size, body_port=args.body, csi_port=args.csi)

    try:
        while True:
            t = time.time()

            if not args.no_demo:
                x = 0.5 + 0.30 * np.sin(t * 0.55)
                y = 0.5 + 0.30 * np.cos(t * 0.41)
                osys.touch(x, y, strength=0.35)

            osys.step(
                drive_body=args.drive,
                closed_loop=not args.no_loop
            )
            osys.save()
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n👋 shutdown")
        osys.save(force=True)
        osys.close()


if __name__ == "__main__":
    main()
