import unittest
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from CNNTripletModel import build_network, build_model, TripletLossLayer

class TestCNNTripletModel(unittest.TestCase):
    # Test that network gets build properly and is type tf
    def proper_return_type(self):
        input_shape = (28,28,1)
        network = build_network(input_shape, embeddingsize=10)
        
        value = type(network)
        expected_value = 'tf'
        
        self.assertTrue(value == expected_value)
        
    # Test that we can build a model with the three networks implemented
    def proper_return_type(self):
        input_shape = (28,28,1)
        model = build_model(input_shape, embeddingsize=10)
        
        value = type(model)
        expected_value = 'tf'
        
        self.assertTrue(value == expected_value)

    # Test the Triplet Loss Layer, that it returns proper distance
    def test_Triplet_Loss_Layer_Functionality(self):
        a_vec = np.zeros(3)
        p_vec = np.ones(3)
        n_vec = np.ones(3) * 2
        
        anchor_input = Input(a_vec.shape, name="anchor_input")
        positive_input = Input(p_vec.shape, name="positive_input")
        negative_input = Input(n_vec.shape, name="negative_input")
        loss_layer = TripletLossLayer(alpha=0.2, name="triplet_loss_layer")(
        [a_vec, p_vec, n_vec])
        network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
        
        value = network_train.predict([a_vec, p_vec, n_vec])
        expected_value = abs(a_vec - p_vec) - abs(a_vec - n_vec)
        
        self.assertTrue(value == expected_value)

    # Test that model has proper input shape
    def test_input_shape(self):
        input_shape = (28,28,1)
        network = build_network(input_shape, embeddingsize=10)
        network_train = build_model(input_shape, network)
        optimizer = Adam(lr=0.00006)
        network_train.compile(loss=None, optimizer=optimizer)
        
        value = network_train.input_shape
        expected_value = input_shape
        
        self.assertTrue(value == expected_value)
    
     # Test that model has proper output shape
    def test_output_shape(self):
        input_shape = (28,28,1)
        network = build_network(input_shape, embeddingsize=10)
        network_train = build_model(input_shape, network)
        optimizer = Adam(lr=0.00006)
        network_train.compile(loss=None, optimizer=optimizer)
        
        value = network_train.output_shape
        expected_value = 10
        
        self.assertTrue(value == expected_value)
    
    # Test that model can be trained and loss changes
    def test_loss_changes(self):
        random.seed(123)
        input_shape = (28,28,1)
        anch = np.random.rand(10,28,28,1)
        pos = np.random.rand(10,28,28,1)
        neg = np.random.rand(10,28,28,1)
        triplets = [anch, pos, neg]
        
        
        network = build_network(input_shape, embeddingsize=10)
        network_train = build_model(input_shape, network)
        optimizer = Adam(lr=0.00006)
        network_train.compile(loss=None, optimizer=optimizer)

        for i in range(3):
            loss = network_train.train_on_batch(triplets, None)

        value = loss[0]- loss[1]
        self.assertTrue(value != 0)
        
    if __name__ == "__main__":
        unittest.main()
