import unittest
import cv2
import matplotlib.pyplot as plt
from ORB import orb_sim, img1, img2

class TestORB(unittest.TestCase):
    # Test if ORB output provides a value between 0-1 inclusive
    def test_general(self):
        value = orb_sim(img1, img2)
        self.assertTrue(value <= 1 and value >= 0)

    #Testing orb_sim with dummy data. Images are identical and should return 1
    def test_dummy_data_identical(self):
        img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        #get path
        img1_inner = cv2.imread(img1_path,0)
        img2_inner = cv2.imread(img2_path,0)
        value = orb_sim(img1_inner,img2_inner)
        self.assertEqual(value, 1)

    #Testing orb_sim with dummy data. Both images are similar and show return a number between 0 and 1
    def test_dummy_data_similar(self):
        img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
        #get path
        img1_inner = cv2.imread(img1_path,0)
        img2_inner = cv2.imread(img2_path,0)
        value = orb_sim(img1_inner,img2_inner)
        self.assertEqual(value,0.8530927835051546)

    #testing orb_sim with dummy data. Both images are opposite and should return 0
    def test_dummy_data_different(self):
        img1_path = "medical-imaging-matching/docs/exploration/Bond3.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        #get path
        img1_inner = cv2.imread(img1_path,0)
        img2_inner = cv2.imread(img2_path,0)
        value = orb_sim(img1_inner,img2_inner) 
        self.assertEqual(value, 0)

    # Check if img1 exists (file path is correct)
    def test_img1_not_none(self):
        self.assertTrue(type(img1) != None)

    # Check if img2 exists (file path is correct)
    def test_img2_not_none(self):
        self.assertTrue(type(img2) != None)

if __name__ == "__main__":
    unittest.main()
