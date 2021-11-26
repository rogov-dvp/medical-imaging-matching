import unittest
import os
import cv2
import numpy as np

from da_numpy import DataAug


class TestDA(unittest.TestCase):
    def test_horizontal_shift(self):
        """
        Tests if the images are shifted horizontally
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        hs_img = da.horizontal_shift()
        result = np.array_equal(img, hs_img)
        self.assertFalse(result)

    def test_vertical_shift(self):
        """
        Tests if the images are shifted vertically
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        vs_img = da.vertical_shift()
        result = np.array_equal(img, vs_img)
        self.assertFalse(result)

    def test_horizontal_flip(self):
        """
        Tests if the image is flipped horizontally
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        flipped_img = da.horizontal_flip()
        result = np.array_equal(img, flipped_img)
        self.assertFalse(result)

    def test_vertical_flip(self):
        """
        Tests if the image is flipped vertically
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        flipped_img = da.vertical_flip()
        result = np.array_equal(img, flipped_img)
        self.assertFalse(result)

    def test_rotation(self):
        """
        Tests if the image is rotated
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        rotated_img = da.rotation(45)
        result = np.array_equal(img, rotated_img)
        self.assertFalse(result)

    def test_blur(self):
        """
        Tests if images are blurred
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        blurred_img = da.blur()
        result = np.array_equal(img, blurred_img)
        self.assertFalse(result)

    def test_zoom(self):
        """
        Tests if images are zoomed
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        zoomed_img = da.zoom(0.5)
        result = np.array_equal(img, zoomed_img)
        self.assertFalse(result)

    def test_horizontal_shift(self):
        """
        Testing if the image has been shifted.
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        hs_img = da.horizontal_shift(0.5)
        result = np.array_equal(img, hs_img)
        self.assertFalse(result)

    def test_vertical_shift(self):
        """
        Testing if the image has been shifted.
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        vs_img = da.vertical_shift(0.5)
        result = np.array_equal(img, vs_img)
        self.assertFalse(result)

    def test_brightness(self):
        """
        Test if image has been brightened.
        """
        img = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        da = DataAug(img)
        b_img = da.brightness(2, 250)
        result = np.array_equal(img, b_img)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
