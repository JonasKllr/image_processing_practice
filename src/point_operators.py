import cv2
import numpy as np
from matplotlib import pyplot as plt


class PointOperators:
    def __init__(self, image: np.ndarray):
        self.image = image

    def plot_histogram(self) -> None:
        plt.hist(self.image.ravel(), 256, [0, 255])
        plt.show()
        plt.close()

    def apply_gamma_operator(self, gamma: float) -> None:
