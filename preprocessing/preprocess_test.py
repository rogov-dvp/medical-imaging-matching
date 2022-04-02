import unittest
import os
import cv2
import numpy as np

from preprocess_data import PreprocessData

pp_filepath = "../test_kaggle_images/processed_images/"
processed_path = "../test_images_kaggle/processed_images/"
unprocessed_images = "../test_images_kaggle/images"


class TestPreprocess(unittest.TestCase):
    def test_check_img(self):
        """
        Testing if it checks for image correctly.
        """
        test = PreprocessData(
            "2016_BC003122_ CC_L.jpg", processed_path, unprocessed_images
        )
        test_case = ["../test_images_kaggle/images/2016_BC003122_ CC_L.jpg"]
        result = test.check_imgs(unprocessed_images)
        self.assertEqual(test_case, result)

    def test_load_image(self):
        """
        Testing loading images
        """
        test_image = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData(
            "2016_BC003122_ CC_L.jpg", processed_path, unprocessed_images
        )
        path = test.check_imgs(unprocessed_images)
        image = test.load_image(path[0])
        result = np.array_equal(test_image, image)
        self.assertTrue(result)

    def test_resize_image(self):
        """
        Testing resizing the images
        """
        test_image = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData(
            "2016_BC003122_ CC_L.jpg", processed_path, unprocessed_images
        )
        resized = test.resize_image(test_image)
        result = np.array_equal(test_image, resized)
        self.assertFalse(result)

    def test_save_image(self):
        """
        Testing that the image is saved.
        """
        test_image = cv2.imread("../test_images_kaggle/images/2016_BC003122_ CC_L.jpg")
        test = PreprocessData(
            "2016_BC003122_ CC_L.jpg", processed_path, unprocessed_images
        )
        test.save_image(test_image)
        test_check = PreprocessData(
            "2016_BC003122_ CC_L.npy", processed_path, unprocessed_images
        )
        check = test_check.check_imgs(processed_path)
        self.assertTrue(len(check) != 0)

    def test_process_image(self):
        """
        Testing the process file method
        """
        test = PreprocessData(
            "2016_BC003122_ CC_L.jpg", processed_path, unprocessed_images
        )
        result = test.process_image()
        # if os.path.exists(result):
        #     os.remove(result)
        self.assertTrue("2016_BC003122_ CC_L.npy" in result)


if __name__ == "__main__":
    unittest.main()
