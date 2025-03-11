import cv2
import numpy as np
import os


class ImagePreprocessor:
    def __init__(self, image: np.ndarray):
        self.image = image

    def crop_to_roi(self) -> None:
        self.image = self.image[490:630, 925:1235]

    def rgb_to_grayscale(self) -> None:
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def rotate(self) -> None:
        self.image = cv2.rotate(self.image, cv2.ROTATE_180)
    
    def get_image(self) -> np.ndarray:
        return self.image


def read_image_from_file(PATH: str) -> np.ndarray:
    return cv2.imread(PATH)


if __name__ == "__main__":
    PATH = "/home/jonas/coding_practice/image_processing_practice/img"
    PATH_TO_IMAGE = os.path.join(PATH, "3d_printer.png")

    raw_image = read_image_from_file(PATH_TO_IMAGE)
    image = ImagePreprocessor(raw_image)
    image.crop_to_roi()
    image.rgb_to_grayscale()
    image.rotate()

    window_name = "Image 3D-printer"
    cv2.imshow(window_name, image.get_image())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
