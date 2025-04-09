import numpy as np


class BaseOperator:

    def __init__(self, image: np.ndarray):
        self.image = image

    def get_image(self) -> np.ndarray:
        return self.image
