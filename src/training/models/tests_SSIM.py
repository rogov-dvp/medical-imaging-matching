import unittest
import matplotlib.pyplot as plt
from SSIM import get_sim, img1, img2


class TestSSIM(unittest.TestCase):
    # Test if SSIM output provides a proper value
    def result(self):
        img1_path = ""
        img2_path = ""
        value = get_sim(img1_path, img2_path)
        self.assertTrue(value <= 1 and value >= 0)

 #Testing orb_sim with dummy data. Images are identical and should return 1
    def dummyDataIdentical(self):
        img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        value = get_sim(img1_path,img2_path)
        self.assertEqual(value, 1)

    #Testing orb_sim with dummy data. Both images are similar and show return a number between 0 and 1
    def dummyDataSimilar(self):
        img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
        value = get_sim(img1_path,img2_path)
        self.assertEqual(value,0.7462383476379398)

    #testing orb_sim with dummy data. Both images are opposite and should return 0
    def dummyDataDifferent(self):
        img1_path = "medical-imaging-matching/docs/exploration/Bond3.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        value = get_sim(img1_path,img2_path)
        self.assertEqual(value, 0)

    # Check if img1 exists (file path is correct)
    def img1NotNone(self):
        self.assertTrue(type(img1) != None)

    # Check if img2 exists (file path is correct)
    def img2NotNone(self):
        self.assertTrue(type(img2) != None)

if __name__ == "__main__":
    unittest.main()
