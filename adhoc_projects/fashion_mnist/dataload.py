'''Load Data Functions written in this Script'''

from zipfile import ZipFile
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataLoader(object):
    def __init__(self):
        self.DIR="./data/"
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):        
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    def create_batches(self, x, y, batch_size):
        DLoader = []
        num_classes = len(np.unique(y))
        num_samples = x.shape[0]
        num_batches = num_samples//batch_size if num_samples//batch_size == 0 else (num_samples//batch_size + 1)
        
        OneEncoder = OneHotEncoder(categories=[range(num_classes)], sparse= False)
        y = OneEncoder.fit_transform(y.reshape(-1,1))
        
        for i in range(num_batches):
            batch_x = (x[i*batch_size : (i+1)*batch_size])/255
            batch_y = y[i*batch_size : (i+1)*batch_size]
            DLoader.append((batch_x, batch_y))
        
        return DLoader