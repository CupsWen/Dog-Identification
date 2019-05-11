#!/usr/bin/python
# coding:utf-8

import os
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68] 

class Vgg16():
    def __init__(self, vgg16_path=None, vgg15_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()
            #print self.data_dict
        if vgg15_path is None:
            vgg15_path = os.path.join(os.getcwd(), "vgg15.npy")
            self.data = np.load(vgg15_path, encoding='latin1').item()
            #print(self.data)

    def forward(self, images):

        rgb_scaled = images * 255.0 
        red, green, blue = tf.split(rgb_scaled,3,3) 
        bgr = tf.concat([     
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],3)
        
        self.conv1_1 = self.conv_layer(bgr, "conv1_1") 
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")
        
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")


        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

        self.pool_shape = self.pool5.get_shape().as_list()
        self.nodes = self.pool_shape[1] * self.pool_shape[2] * self.pool_shape[3]
        self.cups = tf.reshape(self.pool5, [self.pool_shape[0], self.nodes])

        self.w1 = tf.constant(self.data['w1'], name='w1')
        self.b1 = tf.constant(self.data['b1'], name='b1')
        self.y1 = tf.nn.relu(tf.matmul(self.cups, self.w1) + self.b1)

        self.w2 = tf.constant(self.data['w2'], name='w2')
        self.b2 = tf.constant(self.data['b2'], name='b2')
        self.y2 = tf.matmul(self.y1, self.w2) + self.b2
        self.prob = tf.nn.softmax(self.y2, name="prob")

        self.data = None
        
    def conv_layer(self, x, name):
        with tf.variable_scope(name): 
            w = self.get_conv_filter(name) 
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME') 
            conv_biases = self.get_bias(name) 
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases)) 
            return result
    
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")
    
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
    
    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def get_fc_weight(self, name):  
        return tf.constant(self.data_dict[name][0], name="weights")

