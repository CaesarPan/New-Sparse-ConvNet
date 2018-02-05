import numpy as np
import tensorflow as tf
from numpy import *

def _zca(X, U_matrix=None, S_matrix=None, mu=None, flag='train', alpha=1e-5):
    """
    preprocess image dataset by zca
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return:
    """
    if flag == 'train':
        #mu = np.mean(X, axis=0)
        #X = X - mu
        cov = np.cov(X.T)
        U, S, V = np.linalg.svd(cov)
    else:
        #X = X - mu
	U = U_matrix
	S = S_matrix
    x_rot = np.dot(X, U)
    pca_whiten = x_rot / np.sqrt(S + alpha)
    zca_whiten = np.dot(pca_whiten, U.T)
    return zca_whiten, U, S, mu


def _gcn(X, flag=0, scale=55.):
    """
    preprocess image dataset by gcn
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return: the dataset after preprocessing by gcn
    """
    if flag == 0:
	print "1"
	mean = np.mean(X, axis=1)
	X = X - mean[:, np.newaxis]
	contrast = np.sqrt(10. + (X**2).sum(axis=1)) / scale
	contrast[contrast < 1e-8] = 1.
	X = X / contrast[:, np.newaxis]
    else:
	print "3"
        X = X.reshape([-1, 32, 32, 3])
        mu = np.mean(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
        std = np.std(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
	X = X - mu
        X = X / std
	X = X.reshape([-1, 3072])
    return X

def _leakyrelu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x =x - alpha * negative_part
    return x

def weight_variable(shape, name=None):
    initial = tf.random_normal(shape, dtype=tf.float32, stddev=0.05)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
