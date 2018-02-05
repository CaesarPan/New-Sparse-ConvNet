# This script runs the entire pipeline with Sparse ConvNet and on CIFAR10 dataset.
# Written by Caesar

##################################################################################################################################################################
#
#                                                                   Import Section
#
##################################################################################################################################################################

import numpy as np
import tensorflow as tf
from scn_model import SCN_Model

##################################################################################################################################################################
#
#                                                                   Global Constant Definition Section
#
##################################################################################################################################################################

_MODEL_SAVE_PATH = "./models/"
_BATCH_SIZE = 50
_EPOCH_NUM = 250

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

# Real data placeholders
x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 32, 32, 3])

# Dropout keep probability placeholders
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
keep_prob5 = tf.placeholder(tf.float32)
keep_prob6 = tf.placeholder(tf.float32)
keep_prob = {"keep_prob2": keep_prob2, "keep_prob3": keep_prob3, "keep_prob4": keep_prob4, "keep_prob5": keep_prob5, "keep_prob6": keep_prob6}

# Model initialization
model = SCN_Model(x_image, y_, keep_prob)

# Training
print("Training")
saver = tf.train.Saver(max_to_keep=0)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(_EPOCH_NUM):
    epoch_num = epoch + 1				# The actual epoch starts from 1

    # Shuffle training data
    print("Shuffling training data...")
    train_set = np.concatenate((train_x, train_y), axis=1)
    np.random.shuffle(train_set)                        # Note that this function is in-place so the set doesn't need to be explicitly assigned the new value
    train_x = train_set[:, :3072]
    train_y = train_set[:, 3072:]
    print("Shuffling finished.")

    for step in range(train_x.shape[0]/_BATCH_SIZE):
        batch_x = train_x[step*_BATCH_SIZE:(step+1)*_BATCH_SIZE-1]
        batch_y = train_y[step*_BATCH_SIZE:(step+1)*_BATCH_SIZE-1]

        if step % 100 == 0:
            train_accuracy = sess.run(model.accuracy, feed_dict={image: batch_x, label: batch_y, 
                                                                 keep_prob: {"keep_prob2": 1.0, "keep_prob3": 1.0, 
                                                                             "keep_prob4": 1.0, "keep_prob5": 1.0, "keep_prob6": 1.0}})
            print("epoch %d, step %d, training accuracy %g" % (epoch_num, step, train_accuracy))
        
        sess.run(model.optimize, feed_dict={image: batch_x, label: batch_y,
                                            keep_prob: {"keep_prob2": 0.9, "keep_prob3": 0.8, 
                                                        "keep_prob4": 0.7, "keep_prob5": 0.6, "keep_prob6": 0.5}})

    if epoch_num >= (_EPOCH_NUM - 10):           # Only save models for the last 10 epochs
        saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=epoch_num)
        print("Saved checkpoint for epoch %d and training accuracy %g." % (epoch_num, train_accuracy))
   
    print("Training finished.")

for i in range(len(test_x)/_BATCH_SIZE):
    test_batch_x = test_x[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_batch_y = test_y[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_prediction[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1] = sess.run(model.prediction, feed_dict={image: test_batch_x, label: test_batch_y, 
                                                            keep_prob: {"keep_prob2": 1.0, "keep_prob3": 1.0,
                                                                        "keep_prob4": 1.0, "keep_prob5": 1.0, "keep_prob6": 1.0}})

# Test accuracy computation
test_correct_pred = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_y, 1))
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
