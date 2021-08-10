import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from os                             import listdir
from os.path                        import isfile, join, dirname
from random                         import sample
from matplotlib.pyplot              import imshow
from PIL                            import Image
from tensorflow.keras               import Sequential, layers, Model
from tensorflow.keras.callbacks     import ModelCheckpoint, EarlyStopping, LearningRateScheduler

def feed_forward(base_model, image_shape, data_aug=None, preprocess=None, clf_layers=None):
    inputs = tf.keras.Input(shape=image_shape)
    if preprocess is not None:
        x = preprocess(inputs)
        x = data_aug(x)
    else:
        x = data_aug(inputs)
    x = base_model(x)
    outputs = clf_layers(x)
    return inputs, outputs

def scheduler(epochs, lr):
    return lr if epochs<5 else lr * tf.math.exp(-0.1)


def display_augmented(
    dataset, 
    data_aug, 
    n_images=1, 
    n_rows=2, 
    n_cols=10, 
    figsize=(30, 5), 
    preprocess=None, 
    dtype=tf.float32
):
    for image, _ in dataset.shuffle(n_images).take(n_images):
        _image = tf.expand_dims(image[0], 0)

        _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if preprocess is not None:
            if dtype is not None:
                _image = tf.image.convert_image_dtype(_image, dtype)
            _image = preprocess(_image)
        for ax in axes.flatten(): ax.imshow(data_aug(_image)[0])
        plt.axis("off")
        plt.show()
    pass