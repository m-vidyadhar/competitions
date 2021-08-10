from os import listdir

import numpy as np
import pandas as pd
from random import sample

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import tensorflow as tf


class DataLoader(object):
    def __init__(self, images_path, labels_path=None):
        self.images_path    = images_path
        self.labels_path    = labels_path
        self.files          = None
        self.images         = {}
        self.dataset        = None
        pass
    
    def display_samples(self, image_names=None, grid=(5, 10), figsize=(30, 30)):
        image_names = listdir(self.images_path) if image_names is None else image_names
        if (len(image_names) > np.product(grid)):
            image_names = sample(image_names, np.product(grid))
        
        _, axes = plt.subplots(grid[0], grid[1], figsize=figsize)
        images = []
        
        for file in image_names: images.append(np.asarray(Image.open(self.images_path+file)))
        for img, ax in zip(images, axes.flatten()): ax.imshow(img)
        plt.show()
        pass
    
    def load_data(self, files=None):
        train_labels, train_images, test_images  = [], [], []
        self.files = files if files is not None else listdir(self.images_path)
        
        self.images_count = len(self.files)
        
        self.train_df = pd.read_csv(self.labels_path+"train.csv").set_index("image_names")
        
        for file in self.files:
            if file in self.train_df.index.tolist():
                train_images.append(np.asarray(Image.open(self.images_path+file)))
                train_labels.append(self.train_df.loc[file, "emergency_or_not"])
            else:
                test_images.append(np.asarray(Image.open(self.images_path+file)))
        
        self.image_size = train_images[0].shape
        
        self.images["x_train"] = train_images
        self.images["y_train"] = train_labels
        self.images["x_test"] = test_images
        return self.images
    
    def load_tfdataset(self, files=None):
        if self.images == {}:
            self.images = self.load_data(files=files)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.images["x_train"], self.images["y_train"]))
        return self.dataset

    def train_test_split(self, train_split=0.7, batch_size=64, validation=True, buffer_size=None, files=None):
        if self.dataset is None:
            self.dataset = self.load_tfdataset(files=files)
                
        size = len(self.images["y_train"])
        buffer_size = size if buffer_size is None else buffer_size

        self.dataset = self.dataset.shuffle(buffer_size)
        train_dataset = self.dataset.take(int(train_split * size)).batch(batch_size)
        test_dataset = self.dataset.skip(int(train_split * size))

        if validation:
            test_split = (1 - train_split) / 2
            val_dataset = test_dataset.skip(int(test_split * size)).batch(batch_size)
            test_dataset = test_dataset.take(int(test_split * size)).batch(batch_size)
            
            return train_dataset, val_dataset, test_dataset
        
        return train_dataset, test_dataset.batch(batch_size)