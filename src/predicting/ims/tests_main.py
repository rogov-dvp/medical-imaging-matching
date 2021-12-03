import unittest
import os
from main import query


class MainTesting(unittest.TestCase):
    def query_test_nofile(self):
        """
        Testing query method
        """
        result = query("filename")
        expected_result = []
        self.assertEquals(len(expected_result), len(result))


if __name__ == "__main__":
    unittest.main()
