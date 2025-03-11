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