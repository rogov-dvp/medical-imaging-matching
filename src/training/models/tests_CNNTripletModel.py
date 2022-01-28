import unittest 
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from CNNTripletModel import build_network, build_model, TripletLossLayer

class TestCNNTripletModel(unittest.TestCase):
    # Test the Triplet Loss Layer, that it returns proper distance
    def test_Triplet_Loss_Layer_Functionality(self):
        input_shape = (28,28,1)
        anch = np.random.rand(1,28,28,1)
        pos = np.random.rand(1,28,28,1)
        neg = np.random.rand(1,28,28,1)
        triplets = [anch, pos, neg]
        
        network = build_network(input_shape, embeddingsize=10)
        network_train = build_model(input_shape, network)
        optimizer = Adam(lr=0.00006)
        network_train.compile(loss=None, optimizer=optimizer)
        
        value = round(network_train.predict(triplets),5)
        emebed_a = network(anch)
        emebed_p = network(pos)
        emebed_n = network(neg)
        p_dist = np.sum(np.square(emebed_a - emebed_p), axis=-1)
        n_dist = np.sum(np.square(emebed_a - emebed_n), axis=-1)
        expected_value = round(np.sum(np.maximum(p_dist - n_dist + 0.2, 0), axis=0),6)
        
        self.assertTrue(abs(value-expected_value) < 0.0001)

    # Test that model has proper input shape
    def test_input_shape(self):
        input_shape = (28,28,1)
        network = build_network(input_shape, embeddingsize=10)
        network_train = build_model(input_shape, network)
        optimizer = Adam(lr=0.00006)
        network_train.compile(loss=None, optimizer=optimizer)
        
        value = network_train.input_shape
        expected_value = [(None,28,28,1)] * 3
        for i in range(3):
            self.assertTrue(value[i] == expected_value[i])
    
     # Test that model has proper output shape
    def test_output_shape(self):
        input_shape = (28,28,1)
        network = build_network(input_shape, embeddingsize=10)
        
        value = network.output_shape
        expected_value = (None,10)
        
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

        loss = []
        for i in range(3):
             loss.append(network_train.train_on_batch(triplets, None))

        value = loss[0]- loss[1]
        self.assertTrue(value != 0)

    # Test that the loss does not go to zero
    def test_no_zero_loss(self):
        in_tensor = tf.placeholder(tf.float32, (None, 3))
        labels = tf.placeholder(tf.int32, None, 1)
        model = Model(in_tensor, labels)
        sess = tf.Session()
        loss = sess.run(model.loss, feed_dict={
            in_tensor:np.ones(1, 3),
            labels:[[1]]
        })
        assert loss != 0
    
     # Test Encodings are generated
    def test_endocdings_generated(self):
        model = Model(in_tensor, labels)
        value = network_train.input_shape

        self.assertTrue(value != 0)

    
    
     # Test inputs are connected to outputs
   # def test_connection(self):
   #     anch = np.random.rand(10,28,28,1)
   #     pos = np.random.rand(10,28,28,1)
   #     neg = np.random.rand(10,28,28,1)


    
    
    
        
if __name__ == "__main__":
    unittest.main()
