import unittest
import os
import cv2
import numpy as np

from preprocess_data import PreprocessData


class TestPreprocess(unittest.TestCase):
    def test_check_img(self):
        """
        Testing if it checks for image correctly.
        """
        test = PreprocessData("2016_BC003122_ CC_L.jpg")
        test_case = ["test_images_kaggle/images/2016_BC003122_ CC_L.jpg"]
        result = test.check_imgs("test_images_kaggle/images")
        self.assertEqual(test_case, result)

    def test_load_image(self):
        """
        Testing loading images
        """
        test_image = cv2.imread("test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData("2016_BC003122_ CC_L.jpg")
        path = test.check_imgs("test_images_kaggle/images")
        image = test.load_image(path[0])
        result = np.array_equal(test_image, image)
        self.assertTrue(result)

    def test_resize_image(self):
        """
        Testing resizing the images
        """
        test_image = cv2.imread("test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData("2016_BC003122_ CC_L.jpg")
        resized = test.resize_image(test_image)
        result = np.array_equal(test_image, resized)
        self.assertFalse(result)

    def test_save_image(self):
        """
        Testing that the image is saved.
        """
        test_image = cv2.imread("test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData("2016_BC003122_ CC_L.jpg")
        test.save_image(test_image)
        check = test.check_imgs("test_images_kaggle/processed_images")
        if os.path.exists(check[0]):
            os.remove(check[0])
        self.assertTrue(len(check) != 0)

    def test_process_image(self):
        """
        Testing the process file method
        """
        s = "Image saved here: "
        test = PreprocessData("2016_BC003122_ CC_L.jpg")
        result = test.process_image()
        check = test.check_imgs("test_images_kaggle/processed_images")
        if os.path.exists(check[0]):
            os.remove(check[0])
        self.assertTrue(s in result)


if __name__ == "__main__":
    unittest.main()
