import cv2
import numpy as np
from matplotlib import pyplot as plt

class LinearOperators:
    def __init__(self, image: np.ndarray):
        self.image = image

    def get_image(self) -> np.ndarray:
        return self.image
    
    def apply_gaussian_avg_filter(self, kernel_size: list, sigma: int) -> None:
        self.image = cv2.GaussianBlur(self.image, kernel_size, sigma)

    def apply_derivative_sobel(self, orientation: str) -> None:
        if orientation == "horizontal":
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        if orientation == "vertical":
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        
        self.image = np.absolute(self.image)
        self.image = np.uint8(self.image)