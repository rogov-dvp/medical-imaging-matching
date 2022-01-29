import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
  

def train_cnn(train_datagen, training_images, training_labels, validation_datagen, testing_images, testing_labels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

    # Compile Model. 
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Train the Model
    model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32),
                                steps_per_epoch=len(training_images) / 32,
                                epochs=2,
                                validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                                validation_steps=len(testing_images) / 32)

    model.evaluate(testing_images, testing_labels, verbose=0)