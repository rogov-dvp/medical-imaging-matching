#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:18:53 2022

@author: ntecklenburg
"""

import time
import numpy as np
import random
import os

from matplotlib import pyplot as plt

from keras.layers import Input, Conv2D
from keras.models import Sequential
from keras.models import Model
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D, MaxPool2D
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.applications import vgg16

from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle

img_size = 224
path = ''
img_rcc = np.load(path + 'img_rcc_' + str(img_size)+'.npy')
img_lcc = np.load(path + 'img_lcc_' + str(img_size)+'.npy')
img_rmlo = np.load(path + 'img_rmlo_' + str(img_size)+'.npy')
img_lmlo = np.load(path + 'img_lmlo_' + str(img_size)+'.npy')
lab_rcc = np.load(path + 'lab_rcc_' + str(img_size)+'.npy')
lab_lcc = np.load(path + 'lab_lcc_' + str(img_size)+'.npy')
lab_rmlo = np.load(path + 'lab_rmlo_' + str(img_size)+'.npy')
lab_lmlo = np.load(path + 'lab_lmlo_' + str(img_size)+'.npy')


rcc_data_train, rcc_data_test, rcc_labels_train, rcc_labels_test = train_test_split(img_rcc, lab_rcc, test_size=0.2, random_state=42)
lcc_data_train, lcc_data_test, lcc_labels_train, lcc_labels_test = train_test_split(img_lcc, lab_lcc, test_size=0.2, random_state=42)
rmlo_data_train, rmlo_data_test, rmlo_labels_train, rmlo_labels_test = train_test_split(img_rmlo, lab_rmlo, test_size=0.2, random_state=42)
lmlo_data_train, lmlo_data_test, lmlo_labels_train, lmlo_labels_test = train_test_split(img_lmlo, lab_lmlo, test_size=0.2, random_state=42)

data_train = [rcc_data_train, lcc_data_train, rmlo_data_train, lmlo_data_train]
labels_train = [rcc_labels_train, lcc_labels_train, rmlo_labels_train, lmlo_labels_train]
data_test = [rcc_data_test, lcc_data_test, rmlo_data_test, lmlo_data_test]
labels_test = [rcc_labels_test, lcc_labels_test, rmlo_labels_test, lmlo_labels_test]


def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    # model = Sequential()
    # model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
    #                 kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (7,7), activation='relu',
    #                   kernel_initializer=initialize_weights,
    #                   bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                   bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                   bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(Flatten())
    # model.add(Dense(4096, activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


def get_batch(data_train, data_labels, batch_size, pos_neg_relation=0.5):
    h, w, c = data_train[0][0,0].shape
    
    pairs=[np.zeros((batch_size, h, w, c)) for i in range(2)]
    
    cat = random.randint(0,len(data_train)-1)
    targets = np.random.choice([0,1], size=(batch_size,),p=[pos_neg_relation,1-pos_neg_relation])
    
    for i in range(batch_size):
        pos_ind = random.randint(0, len(data_train[cat])-1)
        neg_ind = random.randint(0, len(data_train[cat])-1)
        
        #Pick one random anchor and positive image
        pairs[0][i,:,:,0] = data_train[cat][pos_ind,0,:,:,0]
        pairs[1][i,:,:,0] = data_train[cat][pos_ind,1,:,:,0]
        
        
        if targets[i] == 0:
            #Pick negative image of different patient different from different patient
            while data_labels[cat][neg_ind] == data_labels[cat][pos_ind]:
                neg_ind = random.randint(0, len(data_train[cat])-1)
            rand_ind = random.randint(0,1)
            pairs[1][i,:,:,0] = data_train[cat][neg_ind,rand_ind,:,:,0]
            
        cat += 1
        cat = cat % len(data_train)
        
    return pairs, targets


def get_batch_hard(data_train, data_labels, batch_size, selection_size, model):
    
    pos, targets = get_batch(data_train, data_labels, int(batch_size/2), pos_neg_relation=0)
    
    candidates_neg, targets = get_batch(data_train, data_labels, selection_size, pos_neg_relation=1)
    similarities = model.predict(candidates_neg)
    similarities = np.reshape(similarities, (selection_size))
    ind = np.argsort(similarities)
    ind = np.ndarray.tolist(ind)
    neg = [candidate for _, candidate in sorted(zip(ind,candidates_neg), reverse=True)]
    
    even_ind = list(range(0, 2*pos[0].shape[0], 2))
    odd_ind = list(range(1, 2*pos[0].shape[0], 2))
    
    out_shape = (2*pos[0].shape[0],) + pos[0].shape[1:]
    imgs_0 = np.zeros(out_shape)
    imgs_1 = np.zeros(out_shape)
    
    c = np.random.randint(int(batch_size/4), high=int(batch_size-1), size=int(batch_size/4))
    
    imgs_0[0::2] = pos[0]
    imgs_1[0::2] = pos[1]
    imgs_0[1::4] = neg[0][:int(batch_size/4)]
    imgs_1[1::4] = neg[1][:int(batch_size/4)]
    imgs_0[3::4] = neg[0][int(batch_size/4)]
    imgs_1[3::4] = neg[1][int(batch_size/4)]
    
    pairs = [imgs_0, imgs_1]
    
    targets = np.zeros(2*pos[0].shape[0])
    targets[0::2] = 1
    
    return pairs, targets


def generate(data_train, data_labels, batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(data_train, data_labels, batch_size)
        yield (pairs, targets)
        
        
def make_oneshot_task(data, data_labels, N):
    h, w, c = data[0][0,0].shape
    
    pairs=[np.zeros((N, h, w, c)) for i in range(2)]
    
    cat = random.randint(0,len(data)-1)
    targets = np.zeros(N)
    rand_pos_ind = random.randint(0,N-1)
    targets[rand_pos_ind] = 1
    pos_ind = random.randint(0, len(data[cat])-1)
    
    for i in range(1,N):
        neg_ind = random.randint(0, len(data[cat])-1)
        
        #Pick one random anchor and positive image
        if i==rand_pos_ind:
            pairs[0][i,:,:,0] = data[cat][pos_ind,0,:,:,0]
            pairs[1][i,:,:,0] = data[cat][pos_ind,1,:,:,0]
        else:
            #Pick negative image of different patient different from different patient
            while data_labels[cat][neg_ind] == data_labels[cat][pos_ind]:
                neg_ind = random.randint(0, len(data[cat])-1)
            pairs[0][i,:,:,0] = data[cat][pos_ind,0,:,:,0]
            pairs[1][i,:,:,0] = data[cat][neg_ind,0,:,:,0]
            
    return pairs, targets


def test_oneshot(data, labels, model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(data, labels, N)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct



# Hyper parameters
train_from_scratch = False
evaluate_every = 100 # interval for evaluating on one-shot tasks
batch_size = 32
selection_size = 128 # must be even
n_iter = 10000 # No. of training iterations
N_way = 10 # how many classes for testing one-shot tasks
n_val = 100 # how many one-shot tasks to validate on
best = -1

model_path = "./weights/"

model = get_siamese_model((224,224,1))
model.summary()
optimizer = Adam(0.0000001)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

if not train_from_scratch:
    model.load_weights(os.path.join(model_path, 'weights.h5'))

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
loss_history = []
val_history = []
for i in range(1, n_iter+1):
    (inputs, targets) = get_batch_hard(data_train, labels_train, batch_size, selection_size, model)
    # (inputs, targets) = get_batch(data_train, labels_train, batch_size)
    loss = model.train_on_batch(inputs, targets)
    loss_history.append(loss)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(data_test, labels_test, model, N_way, n_val, verbose=True)
        val_history.append(val_acc)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
            
plt.plot(loss_history)
plt.plot(val_history)