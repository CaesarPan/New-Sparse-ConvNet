# This script constructs a Sparse ConvNet and runs on the CIFAR10 dataset
# Written by Caesar

##################################################################################################################################################################
#
#                                                                   Import Section
#
##################################################################################################################################################################

import pdb
import tensorflow as tf
import numpy as np
from data import get_data_set
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import utils as utl

##################################################################################################################################################################
#
#                                                                   Global Variable Definition Section
#
##################################################################################################################################################################

_MODEL_SAVE_PATH = "./models/"
_BATCH_SIZE = 50
_EPOCH_NUM = 250
_WEIGHT_DECAY = 3e-4

##################################################################################################################################################################
#
#                                                                   Function Definition Section
#
##################################################################################################################################################################

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


##################################################################################################################################################################
#
#                                                                   Command Section
#
##################################################################################################################################################################

# Read in dataset
print("Loading dataset...")
train_x, train_y, train_l = get_data_set(name="train", cifar=10, aug=False)
test_x, test_y, test_l = get_data_set(name="test", cifar=10, aug=False)
print("Dataset has been loaded.")

### Data preprocessing
# GCN
print("Conducting GCN...")
train_x = utl._gcn(train_x)
test_x = utl._gcn(test_x)
print("GCN finished.")

# ZCA
print("Conducting ZCA...")
train_x, U, S, train_mu = utl._zca(train_x, flag="train")
test_x, U, S, test_mu = utl._zca(test_x, U, S, flag="test")
print("ZCA finished.")

# Real data
x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 32, 32, 3])

### Construct the network
# First layer suit (1 layer suit = 1 conv layer + (dropout) + 1 pooling layer + 1 NiN layer)
W_conv1 = weight_variable([3, 3, 3, 300], name="w_conv1")
b_conv1 = bias_variable([300], name="b_conv1")
h_conv1 = utl._leakyrelu(conv2d(x_image, W_conv1) + b_conv1, alpha=0.33)
#h_conv1 = tf.keras.layers.LeakyRelu(conv2d(x_image, W_conv1) + b_conv1, alpha=0.33)
h_pool1 = max_pool_2x2(h_conv1)

W_nin1 = weight_variable([1, 1, 300, 300], name="w_nin1")
b_nin1 = bias_variable([300], name="b_nin1")
h_nin1 = utl._leakyrelu(conv2d(h_pool1, W_nin1) + b_nin1, alpha=0.33)

# Second layer suit
W_conv2 = weight_variable([2, 2, 300, 600], name="w_conv2")
b_conv2 = bias_variable([600], name="b_conv2")

h_conv2 = utl._leakyrelu(conv2d(h_nin1, W_conv2) + b_conv2, alpha=0.33)
keep_prob2 = tf.placeholder(tf.float32)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob2)
h_pool2 = max_pool_2x2(h_conv2_drop)

W_nin2 = weight_variable([1, 1, 600, 600], name="w_nin2")
b_nin2 = bias_variable([600], name="b_nin2")
h_nin2 = utl._leakyrelu(conv2d(h_pool2, W_nin2) + b_nin2, alpha=0.33)

# Third layer suit
W_conv3 = weight_variable([2, 2, 600, 900], name="w_conv3")
b_conv3 = bias_variable([900], name="b_conv3")

h_conv3 = utl._leakyrelu(conv2d(h_nin2, W_conv3) + b_conv3, alpha=0.33)
keep_prob3 = tf.placeholder(tf.float32)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob3)
h_pool3 = max_pool_2x2(h_conv3_drop)

W_nin3 = weight_variable([1, 1, 900, 900], name="w_nin3")
b_nin3 = bias_variable([900], name="b_nin3")
h_nin3 = utl._leakyrelu(conv2d(h_pool3, W_nin3) + b_nin3, alpha=0.33)

# Fourth layer suit
W_conv4 = weight_variable([2, 2, 900, 1200], name="w_conv4")
b_conv4 = bias_variable([1200], name="b_conv4")

h_conv4 = utl._leakyrelu(conv2d(h_nin3, W_conv4) + b_conv4, alpha=0.33)
keep_prob4 = tf.placeholder(tf.float32)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob4)
h_pool4 = max_pool_2x2(h_conv4_drop)

W_nin4 = weight_variable([1, 1, 1200, 1200], name="w_nin4")
b_nin4 = bias_variable([1200], name="b_nin4")
h_nin4 = utl._leakyrelu(conv2d(h_pool4, W_nin4) + b_nin4, alpha=0.33)

# Fifth layer suit
W_conv5 = weight_variable([2, 2, 1200, 1500], name="w_conv5")
b_conv5 = bias_variable([1500], name="b_conv5")

h_conv5 = utl._leakyrelu(conv2d(h_nin4, W_conv5) + b_conv5, alpha=0.33)
keep_prob5 = tf.placeholder(tf.float32)
h_conv5_drop = tf.nn.dropout(h_conv5, keep_prob5)
h_pool5 = max_pool_2x2(h_conv5_drop)

W_nin5 = weight_variable([1, 1, 1500, 1500], name="w_nin5")
b_nin5 = bias_variable([1500], name="b_nin5")
h_nin5 = utl._leakyrelu(conv2d(h_pool5, W_nin5) + b_nin5, alpha=0.33)

# Sixth conv layer
W_conv6 = weight_variable([2, 2, 1500, 1800], name="w_conv6")
b_conv6 = bias_variable([1800], name="b_conv6")

h_conv6 = utl._leakyrelu(conv2d(h_nin5, W_conv6) + b_conv6, alpha=0.33)
keep_prob6 = tf.placeholder(tf.float32)
h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob6)

W_nin6 = weight_variable([1, 1, 1800, 1800], name="w_nin6")
b_nin6 = bias_variable([1800], name="b_nin6")
h_nin6 = utl._leakyrelu(conv2d(h_conv6_drop, W_nin6) + b_nin6, alpha=0.33)

h_nin6_flat = tf.reshape(h_nin6, [-1, 1 * 1 * 1800])

# Softmax classification layer
W_sm = weight_variable([1 * 1 * 1800, 10], name="w_sm")
b_sm = bias_variable([10], name="b_sm")

y_conv = tf.matmul(h_nin6_flat, W_sm) + b_sm
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Weight decay
vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if "b" not in v.name]) * _WEIGHT_DECAY
loss = cross_entropy + l2_loss

## Training step configuration
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99, use_nesterov=True).minimize(loss)    # lr = 0.003*exp(-0.01*epoch)

# Accuracy definition
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training
print("Training...")
saver = tf.train.Saver(max_to_keep=0)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(_EPOCH_NUM):
    epoch_num = epoch + 1				# The actual epoch starts from 1    

    # Shuffle training data
    print("Shuffling training data...")
    train_set = np.concatenate((train_x, train_y), axis=1)
    np.random.shuffle(train_set)  # Note this function is in-place so the set doesn't need to be explicitly assigned the new value
    train_x = train_set[:, :3072]
    train_y = train_set[:, 3072:]
    print("Shuffling finished.")
    for step in range(train_x.shape[0]/_BATCH_SIZE):
        batch_x = train_x[step*_BATCH_SIZE:(step+1)*_BATCH_SIZE-1]
        batch_y = train_y[step*_BATCH_SIZE:(step+1)*_BATCH_SIZE-1]
        if step % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x: batch_x, y_: batch_y, keep_prob2: 1.0, keep_prob3: 1.0,
                                                      keep_prob4: 1.0, keep_prob5: 1.0, keep_prob6: 1.0})
            print("epoch %d, step %d, training accuracy %g" % (epoch_num, step, train_accuracy))

        train_step.run(session=sess,
                       feed_dict={x: batch_x, y_: batch_y, learning_rate: 1e-6*np.exp(-0.01*epoch_num), keep_prob2: 0.9,#learning_rate: 0.0001 * np.exp(-0.01*(i/1000+1)), keep_prob2: 0.9,
                                  keep_prob3: 0.8, keep_prob4: 0.7, keep_prob5: 0.6, keep_prob6: 0.5})

    if epoch_num >= (_EPOCH_NUM-10):           # Only save models for the last 10 epochs
        saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=epoch_num)
        print("Saved checkpoint for epoch %d and training accuracy %g." % (epoch_num, train_accuracy))


        #train_step.run(session=sess,
        #              feed_dict = {x: batch_i_x, y_: batch_i_y, keep_prob2: 0.9, keep_prob3: 0.8, keep_prob4: 0.7, keep_prob5: 0.6, keep_prob6: 0.5})
print("Training finished.")

test_correct_pred = np.zeros(shape=len(test_x), dtype=np.int)
for i in range(len(test_x)/_BATCH_SIZE):
    test_batch_x = test_x[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_batch_y = test_y[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_correct_pred[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1] = correct_prediction.eval(session=sess,
                                                                                 feed_dict={x: test_batch_x, y_: test_batch_y, keep_prob2: 1.0, keep_prob3: 1.0,
                                                                                            keep_prob4: 1.0, keep_prob5: 1.0, keep_prob6: 1.0})

# Test accuracy computation
test_accuracy = test_correct_pred.mean() * 100
correct_num = test_correct_pred.sum()
test_acc_str = "Accuracy on Test Set: {0:.2f}% ({1} / {2}) \n".format(test_accuracy, correct_num, len(test_x))
print(test_acc_str)
#lprint("test accuracy %g" % accuracy.eval(session=sess,
#                                         feed_dict={x: test_x, y_: test_y, keep_prob2: 1.0, keep_prob3: 1.0,
#                                                    keep_prob4: 1.0, keep_prob5: 1.0, keep_prob6: 1.0}))

f = open("test_accuracy.txt", "w")
f.write(test_acc_str)
f.close()
