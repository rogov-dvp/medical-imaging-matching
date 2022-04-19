import unittest
import tensorflow as tf
import numpy as np
from vgg import get_vgg_model

class TestVGG(unittest.TestCase):
    
    def test_input_shape(self):
        input_shape = 256
        embedding_size = 2048
        model = get_vgg_model(input_shape, embedding_size)
        
        inp_shape = model.input_shape
        expected_inp_shape = (None, input_shape, input_shape, 1)
    
        self.assertEqual(inp_shape, expected_inp_shape)


    def test_embedding_size(self):
        input_shape = 256
        embedding_size = 2048
        model = get_vgg_model(input_shape, embedding_size)
        
        out_shape = model.output_shape
        expexcted = (None, embedding_size)
    
        self.assertEqual(out_shape, expexcted)
    
    def test_model_compiles(self):
        input_shape = 256
        embedding_size = 2048
        model = get_vgg_model(input_shape, embedding_size)
        model.compile(tf.keras.optimizers.Adam(0.003))
            
    def test_model_returns_different_embeddings_for_different_images(self):
        img_1 = np.load('../data/img_rcc_256.npy')[0,0,:,:,:]
        img_2 = np.load('../data/img_rcc_256.npy')[-1,0,:,:,:]
        img_1 = img_1[np.newaxis, ...]
        img_2 = img_2[np.newaxis, ...]
        
        input_shape = 256
        embedding_size = 2048
        model = get_vgg_model(input_shape, embedding_size)
        model.compile(tf.keras.optimizers.Adam(0.003))
        
        embed_1 = model.predict(img_1)
        embed_2 = model.predict(img_2)
        
        self.assertFalse((embed_1 == embed_2).all())
        
        
            
if __name__ == '__main__':
    unittest.main()
    