import unittest
import os
from main import query, preprocess, matching_sim


class MainTesting(unittest.TestCase):
    def query_test_nofile(self):
        """
        Testing query method
        """
        result = query("filename")
        expected_result = []
        self.assertEquals(len(expected_result), len(result))

    def test_preprocess(self):
        """
        Test that the function works as expected within the main
        """
        path = preprocess("2016_BC003122_ CC_L.npy")

        self.assertEqual(
            "../test_images_kaggle/processed_images/lcc/2016_BC003122_ CC_L.npy", path
        )

    def test_matching_sim(self):
        """
        Test that this calls upon the model or returns 69.69 currently.
        """
        self.assertEqual(matching_sim(query("filename")), 69.69)


if __name__ == "__main__":
    unittest.main()
