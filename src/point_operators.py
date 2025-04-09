import cv2
import numpy as np
from matplotlib import pyplot as plt

from base_operator import BaseOperator


class PointOperators(BaseOperator):
    
    def plot_histogram(self) -> None:
        plt.hist(self.image.ravel(), 256, [0, 255])
        plt.show()
        plt.close()

    def apply_gamma_operator(self, gamma: float) -> None:
        for i in range(np.shape(self.image)[0]):
            for j in range(np.shape(self.image)[1]):
                self.image[i][j] = ((self.image[i][j] / 255.0) ** gamma) * 255.0

    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    def apply_gamma_operator_faster(self, gamma: float) -> None:
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        self.image = cv2.LUT(self.image, lookUpTable)

    def apply_histogram_equalization(self) -> None:
        self.image = cv2.equalizeHist(self.image)
