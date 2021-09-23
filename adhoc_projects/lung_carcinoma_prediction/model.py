from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import tensorflow as tf

import sys
import csv
import time
from os import listdir


__author__ = "Vidyadhar Mudium"

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



class ConfFilterCNN(object):
    def __init__(self, x, y):
        self.x = tf.reshape(x, shape=[-1, 128, 128, 1])
        self.y = y

        # conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 20])
            b_conv1 = bias_variable([20])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 20, 32])
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([5, 5, 32, 50])
            b_conv3 = bias_variable([50])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)
        
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([5, 5, 50, 64])
            b_conv4 = bias_variable([64])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)
        
        # fc1
        with tf.variable_scope("fc1"):
            shape = int(np.prod(h_pool4.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 128])
            b_fc1 = bias_variable([128])
            h_pool2_flat = tf.reshape(h_pool4, [-1, shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([128, 2])
            b_fc2 = bias_variable([2])
            y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=y_conv))
        self.pred = tf.argmax(y_conv, 1)

        self.norm = tf.norm(y_conv)

        self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.layer = W_fc1 # because we are interested in adjusting the weights of W_fc1

    def setWeights(self, session, weights):
        for v in tf.trainable_variables():
            session.run(v.assign(weights[v.name]))