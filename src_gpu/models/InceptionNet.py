import numpy as np

from keras.layers import Input, Conv2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D, MaxPool2D
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras import backend as K
from tensorflow.keras.optimizers import Adam


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


def get_inception_model(image_shape, embedding_size):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    model = Sequential()
    model.add(Conv2D(input_shape=(image_shape,image_shape,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))#, kernel_regularizer=l2(5e-4)))
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
    model.add(Dense(embedding_size, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    return model