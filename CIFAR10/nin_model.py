import tensorflow as tf


def model(cifar):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    if cifar == 10:
        _NUM_CLASSES = 10
    else:
        _NUM_CLASSES = 100

    # define functions
    def weight_variable(shape, stddev=0.05):
        initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial, name="w")

    def bias_variable(shape):
        initial = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name="b")

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3x3(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
	count_label = tf.placeholder(tf.int32, shape=[_NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32, name='dropout')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    # conv1 layer
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 192])
        b_conv1 = bias_variable([192])
        output = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-1-1
        with tf.name_scope('mlp_1_1'):
            W_MLP11 = weight_variable([1, 1, 192, 160])
            b_MLP11 = bias_variable([160])
            output = tf.nn.relu(conv2d(output, W_MLP11) + b_MLP11)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-1-2
        with tf.name_scope('mlp_1_2'):
            W_MLP12 = weight_variable([1, 1, 160, 96])
            b_MLP12 = bias_variable([96])
            output = tf.nn.relu(conv2d(output, W_MLP12) + b_MLP12)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))

        # Max pooling
        output = max_pool_3x3(output)
        # dropout
        output = tf.nn.dropout(output, keep_prob)

    # conv2 layer
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 96, 192])
        b_conv2 = bias_variable([192])
        output = tf.nn.relu(conv2d(output, W_conv2) + b_conv2)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-2-1
        with tf.name_scope('mlp_2_1'):
            W_MLP21 = weight_variable([1, 1, 192, 192])
            b_MLP21 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP21) + b_MLP21)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-2-2
        with tf.name_scope('mlp_2_2'):
            W_MLP22 = weight_variable([1, 1, 192, 192])
            b_MLP22 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP22) + b_MLP22)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # Max pooling
        output = max_pool_3x3(output)
        #Avg pooling
        #output = tf.nn.avg_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # dropout
        output = tf.nn.dropout(output, keep_prob)

    # conv3 layer
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 192, 192])
        b_conv3 = bias_variable([192])
        output = tf.nn.relu(conv2d(output, W_conv3) + b_conv3)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-2-1
        with tf.name_scope('mlp_3_1'):
            W_MLP31 = weight_variable([1, 1, 192, 192])
            b_MLP31 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP31) + b_MLP31)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-2-2
        with tf.name_scope('mlp_3_2'):
            W_MLP32 = weight_variable([1, 1, 192, _NUM_CLASSES])
            b_MLP32 = bias_variable([_NUM_CLASSES])
            output = tf.nn.relu(conv2d(output, W_MLP32) + b_MLP32)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # global average
        output = tf.nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')

    with tf.name_scope('output'):
        output = tf.reshape(output, [-1, 1 * 1 * _NUM_CLASSES])
        tf.summary.histogram('avg_pool', output)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(output, dimension=1)

    return x, y, output, global_step, y_pred_cls, keep_prob, count_label
