"""
Echo Grid Ultrasonic OS — Real-time Field Visualizer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from echo_grid.core import EchoGridOS


class EchoVisualizer:
    def __init__(self, size: int = 16):
        self.size = size
        self.osys = EchoGridOS(size)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(11, 5))
        self.im1 = self.ax1.imshow(
            np.zeros((size, size)), cmap="viridis", vmin=-1.2, vmax=1.2, animated=True
        )
        self.im2 = self.ax2.imshow(
            np.zeros((size, size)), cmap="plasma", vmin=38000, vmax=42000, animated=True
        )

        self.ax1.set_title("Phase Field φ")
        self.ax2.set_title("Frequency Map (Hz)")
        plt.colorbar(self.im1, ax=self.ax1, fraction=0.046)
        plt.colorbar(self.im2, ax=self.ax2, fraction=0.046)
        self.fig.suptitle("Echo Grid Ultrasonic OS — Live Field", fontsize=13)

    def update(self, frame):
        t = frame * 0.05
        x = 0.5 + 0.35 * np.sin(t * 0.8)
        y = 0.5 + 0.35 * np.cos(t * 0.55)
        self.osys.touch(x, y, strength=0.9)

        field = self.osys.step()
        freq = 40000 + field * 2000

        self.im1.set_array(field)
        self.im2.set_array(freq)
        return [self.im1, self.im2]


def main():
    vis = EchoVisualizer(size=16)
    ani = FuncAnimation(vis.fig, vis.update, interval=40, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
