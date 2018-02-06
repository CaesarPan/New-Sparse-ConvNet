# This script constructs the Sparse ConvNet model
# Written by Caesar

##################################################################################################################################################################
#
#                                                                   Import Section
#
##################################################################################################################################################################

import tensorflow as tf
import numpy as np
import utils as utl
import functools

##################################################################################################################################################################
#
#                                                                   Global Variable Definition Section
#
##################################################################################################################################################################

_WEIGHT_DECAY = 1e-3

##################################################################################################################################################################
#
#                                                                   Function Definition Section
#
##################################################################################################################################################################

def define_scope(function, scope=None):

    """This is a function decorator that performs lazy instantiation as well as scope definition."""

    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    
    return decorator

##################################################################################################################################################################
#
#                                                                   Model Class Definition Section
#
##################################################################################################################################################################

class SCN_Model:

    """This class maps the Sparse ConvNet model and encapsulates several core functions."""

    def __init__(self, image, label, keep_prob2, keep_prob3, keep_prob4, keep_prob5, keep_prob6):

        """
        :param image: train_x
        :param label: train_y
        :param keep_prob:
        """

        self.image = image
        self.label = label
        self.keep_prob2 = keep_prob2
        self.keep_prob3 = keep_prob3
        self.keep_prob4 = keep_prob4
        self.keep_prob5 = keep_prob5
        self.keep_prob6 = keep_prob6

        # The following 3 lines MUST be placed after those direct assignments above
        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        
        """This function runs the forward propagation and output predictions."""
        
        ### Construct the network
        # First layer suit (1 layer suit = 1 conv layer + (dropout) + 1 pooling layer + 1 NiN layer)
        W_conv1 = utl.weight_variable([3, 3, 3, 300], name="w_conv1")
        b_conv1 = utl.bias_variable([300], name="b_conv1")
        h_conv1 = utl._leakyrelu(utl.conv2d(self.image, W_conv1) + b_conv1, alpha=0.33)
        #h_conv1 = tf.keras.layers.LeakyRelu(utl.conv2d(x_image, W_conv1) + b_conv1, alpha=0.33)
        h_pool1 = utl.max_pool_2x2(h_conv1)
        
        W_nin1 = utl.weight_variable([1, 1, 300, 300], name="w_nin1")
        b_nin1 = utl.bias_variable([300], name="b_nin1")
        h_nin1 = utl._leakyrelu(utl.conv2d(h_pool1, W_nin1) + b_nin1, alpha=0.33)
        
        # Second layer suit
        W_conv2 = utl.weight_variable([2, 2, 300, 600], name="w_conv2")
        b_conv2 = utl.bias_variable([600], name="b_conv2")
        
        h_conv2 = utl._leakyrelu(utl.conv2d(h_nin1, W_conv2) + b_conv2, alpha=0.33)
        #self.keep_prob2 = tf.placeholder(tf.float32)
        h_conv2_drop = tf.nn.dropout(h_conv2, self.keep_prob2)
        h_pool2 = utl.max_pool_2x2(h_conv2_drop)
        
        W_nin2 = utl.weight_variable([1, 1, 600, 600], name="w_nin2")
        b_nin2 = utl.bias_variable([600], name="b_nin2")
        h_nin2 = utl._leakyrelu(utl.conv2d(h_pool2, W_nin2) + b_nin2, alpha=0.33)
        
        # Third layer suit
        W_conv3 = utl.weight_variable([2, 2, 600, 900], name="w_conv3")
        b_conv3 = utl.bias_variable([900], name="b_conv3")
        
        h_conv3 = utl._leakyrelu(utl.conv2d(h_nin2, W_conv3) + b_conv3, alpha=0.33)
        #self.keep_prob3 = tf.placeholder(tf.float32)
        h_conv3_drop = tf.nn.dropout(h_conv3, self.keep_prob3)
        h_pool3 = utl.max_pool_2x2(h_conv3_drop)
        
        W_nin3 = utl.weight_variable([1, 1, 900, 900], name="w_nin3")
        b_nin3 = utl.bias_variable([900], name="b_nin3")
        h_nin3 = utl._leakyrelu(utl.conv2d(h_pool3, W_nin3) + b_nin3, alpha=0.33)
        
        # Fourth layer suit
        W_conv4 = utl.weight_variable([2, 2, 900, 1200], name="w_conv4")
        b_conv4 = utl.bias_variable([1200], name="b_conv4")
        
        h_conv4 = utl._leakyrelu(utl.conv2d(h_nin3, W_conv4) + b_conv4, alpha=0.33)
        #self.keep_prob4 = tf.placeholder(tf.float32)
        h_conv4_drop = tf.nn.dropout(h_conv4, self.keep_prob4)
        h_pool4 = utl.max_pool_2x2(h_conv4_drop)
        
        W_nin4 = utl.weight_variable([1, 1, 1200, 1200], name="w_nin4")
        b_nin4 = utl.bias_variable([1200], name="b_nin4")
        h_nin4 = utl._leakyrelu(utl.conv2d(h_pool4, W_nin4) + b_nin4, alpha=0.33)
        
        # Fifth layer suit
        W_conv5 = utl.weight_variable([2, 2, 1200, 1500], name="w_conv5")
        b_conv5 = utl.bias_variable([1500], name="b_conv5")
        
        h_conv5 = utl._leakyrelu(utl.conv2d(h_nin4, W_conv5) + b_conv5, alpha=0.33)
        #self.keep_prob5 = tf.placeholder(tf.float32)
        h_conv5_drop = tf.nn.dropout(h_conv5, self.keep_prob5)
        h_pool5 = utl.max_pool_2x2(h_conv5_drop)
        
        W_nin5 = utl.weight_variable([1, 1, 1500, 1500], name="w_nin5")
        b_nin5 = utl.bias_variable([1500], name="b_nin5")
        h_nin5 = utl._leakyrelu(utl.conv2d(h_pool5, W_nin5) + b_nin5, alpha=0.33)
        
        # Sixth conv layer
        W_conv6 = utl.weight_variable([2, 2, 1500, 1800], name="w_conv6")
        b_conv6 = utl.bias_variable([1800], name="b_conv6")
        
        h_conv6 = utl._leakyrelu(utl.conv2d(h_nin5, W_conv6) + b_conv6, alpha=0.33)
        #self.keep_prob6 = tf.placeholder(tf.float32)
        h_conv6_drop = tf.nn.dropout(h_conv6, self.keep_prob6)
        
        W_nin6 = utl.weight_variable([1, 1, 1800, 1800], name="w_nin6")
        b_nin6 = utl.bias_variable([1800], name="b_nin6")
        h_nin6 = utl._leakyrelu(utl.conv2d(h_conv6_drop, W_nin6) + b_nin6, alpha=0.33)
        
        h_nin6_flat = tf.reshape(h_nin6, [-1, 1 * 1 * 1800])
        
        # Softmax classification layer
        W_sm = utl.weight_variable([1 * 1 * 1800, 10], name="w_sm")
        b_sm = utl.bias_variable([10], name="b_sm")
        
        y_conv = tf.matmul(h_nin6_flat, W_sm) + b_sm
        
        return y_conv

    @define_scope
    def optimize(self):

        """This function computes the loss term, constructs an optimizer and performs optimization over the loss term."""

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.prediction))

        # Weight decay
        vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if "b" not in v.name]) * _WEIGHT_DECAY
        loss = cross_entropy + l2_loss

        ## Training step configuration
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        #learning_rate = tf.placeholder(tf.float32, shape=[])
        #train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99, use_nesterov=True).minimize(loss)    # lr = 0.003*exp(-0.01*epoch)
        return train_step

    @define_scope
    def accuracy(self):

        """This function computes the prediction accuracy of the model."""

        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
