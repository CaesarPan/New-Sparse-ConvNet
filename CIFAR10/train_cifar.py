import numpy as np
import tensorflow as tf
import utils as utl
import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from data import get_data_set
from nin_model import model

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 120000
_EPOCH = 260
_MODEL_SAVE_PATH = "./cifar/10/add_new_loss/noaug/model/"
_TENSORBOARD_SAVE_PATH = "./cifar/10/add_new_loss/noaug/tensorboard"
CEN_VEC = np.diag(np.ones(_CLASS_SIZE)).astype('float32')


x, y, output, global_step, y_pred_cls, keep_prob, c = model(_CLASS_SIZE)
_output = tf.nn.softmax(output)
o_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))



#------------------------------gcn and zcn process begin------------------------------
train_x, train_y, train_l = get_data_set(name="train", cifar=_CLASS_SIZE, aug=False)
test_x, test_y, test_l = get_data_set(name="test", cifar=_CLASS_SIZE, aug=False)
print "start gcn"
train_x = utl._gcn(train_x)
test_x = utl._gcn(test_x)
print "gcn end"
print "start zca"
train_x, U, S, _mean = utl._zca(train_x)
test_x, _, _, _ = utl._zca(test_x, U, S, flag='test')
print "zca end"
#------------------------------gcn and zcn process end--------------------------------


#------------------------------hyper-parameter setting begin----------------------------
weight_decay = 0.0003
in_near_num = 20
out_near_num = 25
alpha = 2e-2
#------------------------------hyper-parameter setting end------------------------------


#-------------------------------new loss term begin--------------------------------------
dis = utl.sim_dis_tensorflow(_output, CEN_VEC)
loss1, loss2 = utl._loss1_loss2(dis, c, _CLASS_SIZE, in_near_num, out_near_num)
l2_loss = alpha*(loss1 - loss2)
loss = o_loss + l2_loss
#loss = o_loss
#-------------------------------new loss term end----------------------------------------


#-------------------------------trainable variable setting begin-------------------------
t_v = [var for var in tf.trainable_variables()]
w_v = [t_v[2*i] for i in range(8)]
b_v = [t_v[2*i+1] for i in range(8)]
last_w = [t_v[16]]
last_b = [t_v[17]]
w_l2 = tf.add_n([tf.nn.l2_loss(t_v[2*i]) for i in range(len(t_v)/2)])
wl2_loss = w_l2 * weight_decay
#-------------------------------trainable variable setting end  -------------------------


#-------------------------------lr setting begin-----------------------------------------
steps_per_epoch = len(train_x) / _BATCH_SIZE
boundaries = [steps_per_epoch * _epoch for _epoch in [200, 230]]
values_1 = [5e-2, 5e-3, 5e-4]
values_2 = [1e-1, 1e-2, 1e-3]
values_3 = [5e-3, 5e-4, 5e-5]
values_4 = [1e-2, 1e-3, 1e-4]
learning_rate_1 = tf.train.piecewise_constant(global_step, boundaries, values_1)
learning_rate_2 = tf.train.piecewise_constant(global_step, boundaries, values_2)
learning_rate_3 = tf.train.piecewise_constant(global_step, boundaries, values_3)
learning_rate_4 = tf.train.piecewise_constant(global_step, boundaries, values_4)
opt1 = tf.train.MomentumOptimizer(learning_rate_1, 0.9, name='Momentum1', use_nesterov=True)
opt2 = tf.train.MomentumOptimizer(learning_rate_2, 0.9, name='Momentum2', use_nesterov=True)
opt3 = tf.train.MomentumOptimizer(learning_rate_3, 0.9, name='Momentum3', use_nesterov=True)
opt4 = tf.train.MomentumOptimizer(learning_rate_4, 0.9, name='Momentum4', use_nesterov=True)
grads = tf.gradients(loss + wl2_loss, tf.trainable_variables())
grads1 = [grads[2*i] for i in range(8)]
grads2 = [grads[2*i+1] for i in range(8)]
grads3 = [grads[16]]
grads4 = [grads[17]]
train_op1 = opt1.apply_gradients(zip(grads1, w_v))
train_op2 = opt2.apply_gradients(zip(grads2, b_v))
train_op3 = opt3.apply_gradients(zip(grads3, last_w))
train_op4 = opt4.apply_gradients(zip(grads4, last_b), global_step=global_step)
optimizer = tf.group(train_op1, train_op2, train_op3, train_op4)
#-------------------------------lr setting end-------------------------------------------

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)
tf.summary.scalar("Loss", loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=0)
sess = tf.Session()
train_writer = tf.summary.FileWriter(_TENSORBOARD_SAVE_PATH, sess.graph)
sess.run(tf.global_variables_initializer())


def train(num_epoch):
    """
    Train CNN
    :param num_epoch: numbers of epoch
    :return:
    """
    global train_x
    global train_y
    
    epoch_size = len(train_x)

    train_acc = []
    test_acc = []
    _loss1 = []
    _loss2 = []
    _wl2_loss = []
    _ce_loss = []
    _loss = []

    for i in range(num_epoch):
        print ('Epoch: %d' % i)

        randidx = np.arange(epoch_size)
        np.random.shuffle(randidx)
        print (epoch_size)

        train_x = train_x[randidx]
        train_y = train_y[randidx]

        if (epoch_size % _BATCH_SIZE == 0):
            num_iterations = epoch_size / _BATCH_SIZE
        else:
            num_iterations = int(epoch_size / _BATCH_SIZE) + 1

        for j in range(num_iterations):
            if (j < num_iterations - 1):
                batch_xs = train_x[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
                batch_ys = train_y[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
            else:
                batch_xs = train_x[j * _BATCH_SIZE:epoch_size]
                batch_ys = train_y[j * _BATCH_SIZE:epoch_size]

	    ###
	    num_label = np.argmax(batch_ys, 1)
	    sort_label = np.argsort(num_label)
	    batch_xs = batch_xs[sort_label]
	    batch_ys = batch_ys[sort_label]
	    c_label = []
	    for i_j in range(_CLASS_SIZE):
		c_label.append(list(num_label).count(i_j))
	    ###

            i_global, _ = sess.run([global_step, optimizer],
                                   feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, c: c_label})

            if (i_global % 10 == 0) or (j == num_iterations - 1):
                b_loss, b_ce_loss, b_wl2_loss, b_loss1, b_loss2, b_acc = sess.run([loss, o_loss, wl2_loss, loss1, loss2, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, c: c_label})
                print "Global step: %d, batch accuracy: %.3f, loss: %.4f, CE loss: %.4f, wl2 loss: %.4f, loss1: %.4f, loss2: %.4f" % (i_global, b_acc, b_loss, b_ce_loss, b_wl2_loss, b_loss1, b_loss2)
                train_acc.append(b_acc)
                _loss.append(b_loss)
                _ce_loss.append(b_ce_loss)
                _wl2_loss.append(b_wl2_loss)
                _loss1.append(b_loss1)
                _loss2.append(b_loss2)

            if (j == num_iterations - 1):
                data_merged, global_1, l_loss, l_ce_loss, l_wl2_loss, l_loss1, l_loss2, l_acc = sess.run([merged, global_step, loss, o_loss, wl2_loss, loss1, loss2, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, c: c_label})
                te_acc = predict_test()

                test_acc.append(te_acc)      
                train_acc.append(l_acc)
                _loss.append(l_loss)
                _ce_loss.append(l_ce_loss)
                _wl2_loss.append(l_wl2_loss)
                _loss1.append(l_loss1)
                _loss2.append(l_loss2)

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy/test", simple_value=te_acc)])
                train_writer.add_summary(data_merged, global_1)
                train_writer.add_summary(summary, global_1)

        if i>=200:
            saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")
    return train_acc, test_acc, _loss, _ce_loss, _wl2_loss, _loss1, _loss2 


def predict_test(show_confusion_matrix=False):
    """
    Make prediction for all images in test_x
    :param show_confusion_matrix: default false
    :return: accuracy
    """
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc


if _ITERATION != 0:
    time1 = datetime.datetime.now()
    train_acc, test_acc, _loss, _ce_loss, _wl2_loss, _loss1, _loss2 = train(_EPOCH)
    time2 = datetime.datetime.now()
    duration = time2 - time1
    print "duration:", duration
    print "the best test_acc: %.3f, in epoch %d" %(np.max(test_acc), np.argmax(test_acc))

    np.save("./cifar/10/add_new_loss/noaug/train_acc.npy", train_acc)
    np.save("./cifar/10/add_new_loss/noaug/test_acc.npy", test_acc)
    np.save("./cifar/10/add_new_loss/noaug/loss.npy", _loss)
    np.save("./cifar/10/add_new_loss/noaug/ce_loss.npy", _ce_loss)
    np.save("./cifar/10/add_new_loss/noaug/wl2_loss.npy", _wl2_loss)
    np.save("./cifar/10/add_new_loss/noaug/loss1.npy", _loss1)
    np.save("./cifar/10/add_new_loss/noaug/loss2.npy", _loss2)

    fig = plt.figure()
    ax = fig.add_subplot(321)
    x = range(len(_loss))
    y = np.array(_loss).reshape([len(_loss)])
    plt.plot(x,y)

    ax = fig.add_subplot(322)
    x = range(len(_ce_loss))
    y = np.array(_ce_loss).reshape([len(_ce_loss)])
    plt.plot(x,y)

    ax = fig.add_subplot(323)
    x = range(len(_wl2_loss))
    y = np.array(_wl2_loss).reshape([len(_wl2_loss)])
    plt.plot(x,y)

    ax = fig.add_subplot(324)
    x = range(len(_loss1))
    y = np.array(_loss1).reshape([len(_loss1)])
    plt.plot(x,y)

    ax = fig.add_subplot(325)
    x = range(len(_loss2))
    y = np.array(_loss2).reshape([len(_loss2)])
    plt.plot(x,y)

    plt.show()

sess.close()

