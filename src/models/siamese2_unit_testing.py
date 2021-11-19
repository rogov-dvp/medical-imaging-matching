# Testing: Make sure model grabs image or is grabbing correct image
# each component doing what it is meant too
import unittest
import matplotlib.pyplot as plt
from siamese2 import SiameseModel

class TestSiamese2 (unittest.TestCase):
    def preprocess_image(self):
         """
         Tests images correctly load 
         """
         #self.assertEqual()
    def preprocess_triplets(self):
         """
         Tests loading of 3 image filenames 
         """
        # self.assertEqual()
    def visualize(self):
         """
         Tests vizualization of triplets
         """
         #self.assertEqual()
    def test_train_step(self):
         """
         Tests that gradients are being applied to model
         """
        # self.assertEqual()
    def metrics(self):
         """
         Tests that metrics are being listed
         """
        # self.assertEqual()
    def _compute_loss(self):
         """
         Tests compution of triplet loss 
         """ 
        # self.assertEqual()  










if __name__ == "__main__":
    unittest.main()