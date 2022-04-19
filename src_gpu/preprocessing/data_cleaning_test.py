import unittest
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data_cleaning import DataClean

class TestCleaning(unittest.TestCase):
    def test_read_data (self):
         """
        Tests if correct files are being read
         """
         PATH = 'test_images_kaggle/images'
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print("File exists and is readable")
else:
    print("Either the file is missing or not readable")
       
    
    def test_split_data (self):
         """
        Tests that the split between non-numeric and numeric works.
         """
         df = pd.read_csv("../test_images_kaggle/images")
         result = df.select_dtypes(include=[np.number])
         self.assertFalse(result)
    
    
    # see that input is not null
    def test_missing_data (self):
         """
        Using dummy data, ensure that the missing values are found and dealt with.
         """
        img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
        img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
        result = pc_missing(img1_path,img2_path)
        self.assertFalse(result,None)
         
     
     def test_output_data(self):
         """
        Tests for the expected output when reading the file 
         """
        test = DataClean("2016_BC003122_ CC_L.jpg")
        test_case = ["test_images_kaggle/images/2016_BC003122_ CC_L.jpg"]
        result = test.check_imgs("test_images_kaggle/images")
        self.assertEqual(test_case, result)


if __name__ == "__main__":
    unittest.main()
