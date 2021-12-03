import unittest
import matplotlib.pyplot as plt
import numpy as np
from BatchBuilder import get_batch_random

class TestORB(unittest.TestCase):
    # Test if get_batch random returns proper shape of np.array
    def result(self):
        data_train = np.ones((2,2,28,28,1))
        data_train[1] = data_train[1] * 2
        data_labels = np.ones(2)
        data_labels[1] = 2
        
        value = get_batch_random(data_train, data_labels, 10)[0]
        expected_value = (10,28,28,1)
        
        self.assertTrue(value == expected_value)
        
    # Test if get_batch random returns proper shape of list
    def result(self):
        data_train = np.ones((2,2,28,28,1))
        data_train[1] = data_train[1] * 2
        data_labels = np.ones(2)
        data_labels[1] = 2
        
        value = len(get_batch_random(data_train, data_labels, 10))
        expected_value = 3
        
        self.assertTrue(value == expected_value)

    # Testing that get_batch random returns empty array if only one class
    def dummyDataIdentical(self):
        "Could not find negative in 30 tries please check the data and rerun"
        data_train = np.ones((2,2,28,28,1))
        data_labels = np.ones(2)
        
        value = get_batch_random(data_train, data_labels, 10)
        expected_value = "Could not find negative in 30 tries please check the data and rerun"
        
        self.assertTrue(value == expected_value)
        
    # Testing that get_batch does not mix up positive and negative
    def result(self):
        data_train = np.ones((2,2,28,28,1))
        data_train[1] = data_train[1] * 2
        data_labels = np.ones(2)
        data_labels[1] = 2
        
        value_p = get_batch_random(data_train, data_labels, 1)[1][0,0,0,0]
        value_n = get_batch_random(data_train, data_labels, 1)[2][0,0,0,0]
        
        self.assertTrue(value_p != value_n)
    
    if __name__ == "__main__":
        unittest.main()
