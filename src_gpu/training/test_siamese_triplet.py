import sys
sys.path.append('../')

import unittest
import numpy as np
from tensorflow.keras.optimizers import Adam

from SiameseTriplet import build_siamese_triplet_network, SiameseModel
from models.vgg import get_vgg_model

class TestSiameseTripletModel(unittest.TestCase):    
    
    def test_output_for_different_pos_neg_unequal_05(self):
        # build triplet
        img_1 = np.load('../data/img_rcc_256.npy')[0,0,:,:,:]
        img_2 = np.load('../data/img_rcc_256.npy')[0,1,:,:,:]
        img_3 = np.load('../data/img_rcc_256.npy')[-1,0,:,:,:]
        img_1 = img_1[np.newaxis, ...]
        img_2 = img_2[np.newaxis, ...]
        img_3 = img_3[np.newaxis, ...]
        triplet = [img_1, img_2, img_3]
        
        # 0.5 because of margin used in the model
        input_shape = 256
        embeddingsize = 1024
        network =  get_vgg_model(input_shape, embeddingsize)
        siamese_network = build_siamese_triplet_network((input_shape,input_shape,1), network)
        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=Adam(learning_rate=0.00001))
        
        pred = siamese_model._compute_loss(triplet).numpy()[0]
        self.assertFalse(pred == 0.5)
        
        
    def test_output_for_same_pos_neg_is_05(self):
        # build triplet
        img_1 = np.load('../data/img_rcc_256.npy')[0,0,:,:,:]
        img_2 = np.load('../data/img_rcc_256.npy')[0,1,:,:,:]
        img_3 = np.load('../data/img_rcc_256.npy')[0,1,:,:,:]
        img_1 = img_1[np.newaxis, ...]
        img_2 = img_2[np.newaxis, ...]
        img_3 = img_3[np.newaxis, ...]
        triplet = [img_1, img_2, img_3]
        
        # 0.5 because of margin used in the model
        input_shape = 256
        embeddingsize = 1024
        network =  get_vgg_model(input_shape, embeddingsize)
        siamese_network = build_siamese_triplet_network((input_shape,input_shape,1), network)
        siamese_model = SiameseModel(siamese_network)
        siamese_model.compile(optimizer=Adam(learning_rate=0.00001))
        
        pred = siamese_model._compute_loss(triplet).numpy()[0]
        self.assertTrue(pred == 0.5)
        
if __name__ == '__main__':
    unittest.main()
