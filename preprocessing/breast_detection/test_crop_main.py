import unittest
import cv2
import numpy as np

from crop_main_file import set_image, crop_breasts

class TestCropMain(unittest.TestCase):
    def test_crop_breasts(self):
        """
        Testing if adding image to this method will adding image.
        """
        img_test = cv2.imread(
            "../../test_images_kaggle/images/2017_BC015902_ CC_L.jpg")
        img_test_np = np.asarray([img_test])
        result = crop_breasts(img_test_np)
        self.assertIsNotNone(result)

    def test_set_image(self):
        """
        Test if image is loaded and not return None
        """
        img = set_image("../../test_images_kaggle/images/2017_BC015902_ CC_L.jpg")
        self.assertTrue(img != None)


if __name__ == "__main__":
    unittest.main()
