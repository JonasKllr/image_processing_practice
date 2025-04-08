import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

from image_preprocessor import ImagePreprocessor, read_image_from_file
from point_operators import PointOperators
from linear_operators import LinearOperators

PATH = "../img"
PATH_TO_IMAGE = os.path.join(PATH, "3d_printer.png")

raw_image = read_image_from_file(PATH_TO_IMAGE)
image = ImagePreprocessor(raw_image)
image.crop_to_roi()
image.rgb_to_grayscale()
image.rotate()

image_linar_operations = LinearOperators(image.get_image())
image_linar_operations.apply_derivative_sobel("horizontal")
plt.imshow(image_linar_operations.get_image(), cmap="gray")
plt.show()
plt.close()

# image_linar_operations = LinearOperators(image.get_image())
# image_linar_operations.apply_derivative_sobel("vertical")
# plt.imshow(image_linar_operations.get_image(), cmap="gray")
# plt.show()
# plt.close()

filename = "image_derivative_horizontal_2.png"
PATH_TO_SAVE = os.path.join(PATH, filename)
cv2.imwrite(PATH_TO_SAVE, image_linar_operations.get_image())
