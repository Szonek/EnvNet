import cv2
import numpy as np

im = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)
print(type(im))

class PictureLoader:
    def __init__(self, path_to_file):
        self.im = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)