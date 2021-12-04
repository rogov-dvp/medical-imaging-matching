import unittest
import matplotlib.pyplot as plt
import numpy as np
from BatchBuilder import get_batch_random

class TestBatchBuilder(unittest.TestCase):
    # Test if get_batch random returns proper shape of np.array
    def test_proper_return_shape(self):
        data_train = [np.ones((2,2,28,28,1))]
        data_train[0][1] = data_train[0][1] * 2
        data_labels = [np.asarray([1,2])]
        
        value = get_batch_random(data_train, data_labels, 10)[0].shape
        expected_value = (10,28,28,1)
        
        self.assertTrue(value == expected_value)
        
    # Test if get_batch random returns proper shape of list
    def test_proper_length_of_output(self):
        data_train = [np.ones((2,2,28,28,1))]
        data_train[0][1] = data_train[0][1] * 2
        data_labels = [np.ones(2)]
        data_labels[0][1] = 2
        
        value = len(get_batch_random(data_train, data_labels, 10))
        expected_value = 3
        
        self.assertTrue(value == expected_value)

    # Testing that get_batch random returns empty array if only one class
    def test_throws_error_if_only_class(self):
        data_train = [np.ones((2,2,28,28,1))]
        data_labels = [np.ones(2)]
        
        value = get_batch_random(data_train, data_labels, 10)
        expected_value = "Could not find negative in 30 tries please check the data and rerun"
        
        self.assertTrue(value == expected_value)
        
    # Testing that get_batch does not mix up positive and negative
    def test_keeps_positive_and_negative_seperate(self):
        data_train = [np.ones((2,2,4,4,1))]
        data_train[0][1] = np.zeros((2,4,4,1))
        data_labels = [np.asarray([1,2])]
        
        batch = get_batch_random(data_train, data_labels, 1)
        value_p = batch[1][0,0,0,0]
        value_n = batch[2][0,0,0,0]
        
        self.assertTrue(value_p != value_n)
    
if __name__ == "__main__":
    unittest.main()
