import cv2
from ORB import orb_sim, img1, img2
from SSIM import get_sim, img1, img2
# CNNTripletModel
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

from CNNTripletModel import build_network, build_model, TripletLossLayer

# Test the Triplet Loss Layer, that it returns proper distance
def test_Triplet_Loss_Layer_Functionality(regtest):
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
    print(value,file=regtest)

# Test that model has proper input shape
def test_input_shape(regtest):
    input_shape = (28,28,1)
    network = build_network(input_shape, embeddingsize=10)
    network_train = build_model(input_shape, network)
    optimizer = Adam(lr=0.00006)
    network_train.compile(loss=None, optimizer=optimizer)
    
    value = network_train.input_shape
    print(value,file=regtest)

    # Test that model has proper output shape
def test_output_shape(regtest):
    input_shape = (28,28,1)
    network = build_network(input_shape, embeddingsize=10)
    
    value = network.output_shape
    print(value,file=regtest)

# Test that model can be trained and loss changes
def test_loss_changes(regtest):
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
    print(value,file=regtest)


# BatchBuilder
def test_proper_return_shape(regtest):
    data_train = [np.ones((2,2,28,28,1))]
    data_train[0][1] = data_train[0][1] * 2
    data_labels = [np.asarray([1,2])]
    
    value = get_batch_random(data_train, data_labels, 10)[0].shape
    print(value,file=regtest)
    
# Test if get_batch random returns proper shape of list
def test_proper_length_of_output(regtest):
    data_train = [np.ones((2,2,28,28,1))]
    data_train[0][1] = data_train[0][1] * 2
    data_labels = [np.ones(2)]
    data_labels[0][1] = 2
    
    value = len(get_batch_random(data_train, data_labels, 10))
    print(value,file=regtest)

# Testing that get_batch random returns empty array if only one class
def test_throws_error_if_only_class(regtest):
    data_train = [np.ones((2,2,28,28,1))]
    data_labels = [np.ones(2)]
    
    value = get_batch_random(data_train, data_labels, 10)
    print(value,file=regtest)
    
# Testing that get_batch does not mix up positive and negative
def test_keeps_positive_and_negative_seperate(regtest):
    data_train = [np.ones((2,2,4,4,1))]
    data_train[0][1] = np.zeros((2,4,4,1))
    data_labels = [np.asarray([1,2])]
    
    batch = get_batch_random(data_train, data_labels, 1)
    value_p = batch[1][0,0,0,0]
    value_n = batch[2][0,0,0,0]
    print(value_p != value_n,file=regtest)






# ORB REGRESSION TESTING
def test_general(regtest):
    value = orb_sim(img1, img2)
    print(value,file=regtest)   #Write the regression test into file.

#Testing orb_sim with dummy data. Images are identical and should return 1
def test_dummy_data_identical(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner)
    print(value,file=regtest)

#Testing orb_sim with dummy data. Both images are similar and show return a number between 0 and 1
def test_dummy_data_similar(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner)
    print(value,file=regtest)   

#testing orb_sim with dummy data. Both images are opposite and should return 0
def test_dummy_data_different(regtest):
    img1_path = "medical-imaging-matching/docs/exploration/Bond3.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner) 
    print(value,file=regtest)

# Check if img1 exists (file path is correct)
def test_img1_not_none(regtest):
    print(type(img1) != None,file=regtest)

# Check if img2 exists (file path is correct)
def test_img2_not_none(regtest):
    print(type(img2) != None,file=regtest)

# SSIM REGRESSION TESTING
# Test if SSIM output provides a proper value
def test_general(regtest):
    img1_path = ""
    img2_path = ""
    value = get_sim(img1_path, img2_path)
    print(value,file=regtest)

#Testing orb_sim with dummy data. Images are identical and should return 1
def test_dummy_data_identical(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = get_sim(img1_inner,img2_inner)
    print(value,file=regtest)

#Testing orb_sim with dummy data. Both images are similar and show return a number between 0 and 1
def test_dummy_data_similar(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = get_sim(img1_inner,img2_inner)
    print(value,file=regtest)

#testing orb_sim with dummy data. Both images are opposite and should return 0
def test_dummy_data_different(regtest):
    img1_path = "medical-imaging-matching/docs/exploration/Bond3.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = get_sim(img1_inner,img2_inner)
    print(value,file=regtest)

# Check if img1 exists (file path is correct)
def test_img1_not_none(regtest):
    print(type(img1) != None,file=regtest)

# Check if img2 exists (file path is correct)
def test_img2_not_none(regtest):
    print(type(img2) != None,file=regtest)