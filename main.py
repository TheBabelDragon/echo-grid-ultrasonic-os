#!/usr/bin/env python3
"""
Echo Grid Ultrasonic OS — Entry Point
"""

import time
import numpy as np
from echo_grid.core import EchoGridOS


def main():
    print("🚀 Echo Grid Ultrasonic OS v1.0 starting...")
    osys = EchoGridOS(size=16)

    try:
        while True:
            t = time.time()
            # Synthetic moving excitation (demo)
            x = 0.5 + 0.35 * np.sin(t * 0.7)
            y = 0.5 + 0.35 * np.cos(t * 0.5)
            osys.touch(x, y, strength=0.85)

            field = osys.step()

            # Periodic savepoint every ~5 seconds
            if int(t) % 5 == 0:
                osys.save()

            time.sleep(0.016)  # ~60 Hz control loop

    except KeyboardInterrupt:
        print("\n👋 Echo Grid OS shutting down. Final state saved.")
        osys.save()


if __name__ == "__main__":
    main()
