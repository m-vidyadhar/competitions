'''CNN Architecture code is included in this file'''

import numpy as np
import tensorflow as tf

data_aug = 1

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype = tf.float32, stddev = 1e-1)
    return tf.get_variable('weights', shape, initializer = initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(.0)
    return tf.get_variable('bias', shape, initializer = initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class CNN(object):
    def __init__(self, x, y, args = None):
        self.x = tf.reshape(x, (-1, 28, 28, 1))
        self.y = y
        padding = tf.constant([[0, 0],[1, 1,], [1, 1], [0,0]])
        
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            input_x = tf.pad(self.x, paddings=padding)
            h_conv1 = tf.nn.relu(conv2d(input_x, W_conv1) + b_conv1)
            h_pool1 = max_pool2d(h_conv1)
        
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_poo11_p = tf.pad(h_pool1, paddings=padding)
            h_conv2 = tf.nn.relu(conv2d(h_poo11_p, W_conv2) + b_conv2)
            h_pool2 = max_pool2d(h_conv2)
        
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([5, 5, 64, 128])
            b_conv3 = bias_variable([128])
            h_poo12_p = tf.pad(h_pool2, paddings=padding)
            h_conv3 = tf.nn.relu(conv2d(h_poo12_p, W_conv3) + b_conv3)
        
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([5, 5, 128, 128])
            b_conv4 = bias_variable([128])
            h_conv3_p = tf.pad(h_conv3, paddings=padding)
            h_conv4 = tf.nn.relu(conv2d(h_conv3_p, W_conv4) + b_conv4)
            h_pool3 = max_pool2d(h_conv4)
        
        with tf.variable_scope('fc1'):
            shape = int(np.prod(h_pool3.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 256])
            b_fc1 = bias_variable([256])
            h_pool1_flat = tf.reshape(h_pool3, [-1, shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        
        with tf.variable_scope('fc2'):
            W_fc2 = weight_variable([256, 256])
            b_fc2 = bias_variable([256])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        
        with tf.variable_scope('logits_layer'):
            W_fc3 = weight_variable([256, 10])
            b_fc3 = bias_variable([10])
            logits = tf.matmul(h_fc2, W_fc3) + b_fc3
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
        self.predictions = {
            'classes' : tf.argmax(logits, axis = 1),
            'prob' : tf.nn.softmax(logits, name='softmax_tensor')
        }
        
        self.correct_predicted = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(self.y, axis = 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predicted, tf.float32))

class CNN3(object):
    def __init__(self, x, y, args = None):
        self.x = tf.reshape(x, (-1, 28, 28, 1))
        self.y = y
        padding = tf.constant([[0, 0],[1, 1,], [1, 1], [0,0]])
        
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 1, 32])
            b_conv1 = bias_variable([32])
            input_x = tf.pad(self.x, paddings=padding)
            h_conv1 = tf.nn.relu(conv2d(input_x, W_conv1) + b_conv1)
            h_pool1 = max_pool2d(h_conv1)
        
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 32, 64])
            b_conv2 = bias_variable([64])
            h_poo11_p = tf.pad(h_pool1, paddings=padding)
            h_conv2 = tf.nn.relu(conv2d(h_poo11_p, W_conv2) + b_conv2)
            h_pool2 = max_pool2d(h_conv2)
        
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 64, 128])
            b_conv3 = bias_variable([128])
            h_poo12_p = tf.pad(h_pool2, paddings=padding)
            h_conv3 = tf.nn.relu(conv2d(h_poo12_p, W_conv3) + b_conv3)
        
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([3, 3, 128, 128])
            b_conv4 = bias_variable([128])
            h_conv3_p = tf.pad(h_conv3, paddings=padding)
            h_conv4 = tf.nn.relu(conv2d(h_conv3_p, W_conv4) + b_conv4)
            h_pool3 = max_pool2d(h_conv4)
        
        with tf.variable_scope('fc1'):
            shape = int(np.prod(h_pool3.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 256])
            b_fc1 = bias_variable([256])
            h_pool1_flat = tf.reshape(h_pool3, [-1, shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        
        with tf.variable_scope('fc2'):
            W_fc2 = weight_variable([256, 256])
            b_fc2 = bias_variable([256])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        
        with tf.variable_scope('logits_layer'):
            W_fc3 = weight_variable([256, 10])
            b_fc3 = bias_variable([10])
            logits = tf.matmul(h_fc2, W_fc3) + b_fc3
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
        self.predictions = {
            'classes' : tf.argmax(logits, axis = 1),
            'prob' : tf.nn.softmax(logits, name='softmax_tensor')
        }
        
        self.correct_predicted = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(self.y, axis = 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predicted, tf.float32))