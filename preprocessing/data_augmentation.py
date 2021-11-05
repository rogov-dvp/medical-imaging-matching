import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

import os
import PIL
import PIL.Image
import pathlib


def load_images(data_dir):
    """
    Load images and split into datasets for preprocessing.
    """
    batch_size = 32
    img_height = 4000
    img_width = 4000

    # load data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    return [train_ds, val_ds]


path = "../test_images_kaggle/images"
train_ds, val_ds = load_images(path)

for image_batch in train_ds:
    print(image_batch.shape)
