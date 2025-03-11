import cv2
import os
from matplotlib import pyplot as plt

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

plt.imshow(cv2.cvtColor(image.get_image(), cv2.IMREAD_GRAYSCALE))
plt.show()
plt.close()

image_point_operations = PointOperators(image.get_image())
image_point_operations.plot_histogram()
image_point_operations.apply_histogram_equalization()
image_point_operations.plot_histogram()
plt.imshow(cv2.cvtColor(image_point_operations.get_image(), cv2.IMREAD_GRAYSCALE))
plt.show()
plt.close()

image_linar_operations = LinearOperators(image.get_image())
image_linar_operations.apply_gaussian_avg_filter((9,9), 3)
plt.imshow(cv2.cvtColor(image_linar_operations.get_image(), cv2.IMREAD_GRAYSCALE))
plt.show()
plt.close()
