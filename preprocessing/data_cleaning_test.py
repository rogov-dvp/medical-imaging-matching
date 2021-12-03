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
         self.assert
    def test_missing_data (self):
         """
        Using dummy data, ensure that the missing values are found and dealt with.
         """
         self.assert
         
     def test_output_data(self):
         """
        Tests for the expected output when reading the file 
         """
         self.assert     









if __name__ == "__main__":
    unittest.main()
