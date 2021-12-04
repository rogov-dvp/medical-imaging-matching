# Testing: Make sure model grabs image or is grabbing correct image
# each component doing what it is meant too
import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import tensorflow as tf

from siamese2 import SiameseModel


class TestSiamese2(unittest.TestCase):

    def test_preprocess_image(self):
        """
         Tests images correctly load
         """
        # self.assertEqual() 

    def test_preprocess_triplets(self):
        """
         Tests loading of 3 image filenames
         """

    # self.assertEqual()
    def test_visualize(self):
        """
         Tests vizualization of triplets
         """
        # self.assertEqual()

    def test_train_step(self):
        """
         Tests that gradients are being applied to model
         """

    # self.assertEqual()
   # def test_metrics(self):
    #     """
     #    Tests that metrics are being listed
      #   """

    # self.assertEqual()
    # Triplet loss layer
    def test_compute_loss(self):
        """
         Tests compution of triplet loss
         """
        triplets = [anch, pos, neg] 

        self.assertTrue(abs(value-expected_value) < 0.0001)
    #randomly generate a pair of vectors
    def test_cosine_similarity(self):
        """
        Tests that the return type is what we expect and that based on the dummy data we pass in the calculation is correct.
        """
        img1 = ''
        img2 = ''
        result = 1 - spatial.distance.cosine(img1, img2)
        self.assert
    
    
    def test_euclid_distance():
        """
        Tests that the return type is what we expect and that based on the dummy data we pass in the calculation is correct.
        """    
        self.assert

 
    def test_siameseModel(self):
        img1_path = ""
        img2_path = ""
        result = get_sim(img1_path, img2_path)
        self.assertTrue(result <= 1 and result >= 0)  
       


if __name__ == "__main__":
    unittest.main()
