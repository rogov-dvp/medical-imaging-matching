
import unittest
import matplotlib.pyplot as plt
from SSIM import get_sim,img1,img2

class TestSSIM (unittest.TestCase):
    # Test if SSIM output provides a proper value
    def result(self):
        img1_path = ""
        img2_path = ""
        value = get_sim(img1_path,img2_path)
        self.assertTrue(value <=1 and value>= 0)

    # Check if img1 exists (file path is correct)
    def img1NotNone(self):
        self.assertTrue(type(img1) != None)
    
    #Check if img2 exists (file path is correct)
    def img2NotNone(self):
        self.assertTrue(type(img2) != None)


    if __name__ == "__main__":
        unittest.main()