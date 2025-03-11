import cv2
import os

from image_preprocessor import ImagePreprocessor, read_image_from_file
from point_operators import PointOperators

PATH = "/home/jonas/coding_practice/image_processing_practice/img"
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
