import tensorflow as tf

with tf.name_scope('variable'):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr = tf.Variable(0.001, dtype=tf.float32, name='lr')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 2000], name='x_input')
    y = tf.placeholder(tf.float32, [None], name='y_input')
    Y = tf.one_hot(indices=tf.cast(y, tf.int32), depth=13, name='y_onehot')

def captcha_nn():
    """ The network of level1. """
    with tf.name_scope('L1'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([2000, 500], stddev=0.1), name='W1')
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([500])+0.1, name='b1')
        with tf.name_scope('tanh'):
            L1 = tf.nn.tanh(tf.matmul(x, W1)+b1, name='tanh')
        with tf.name_scope('L1_drop'):
            L1_drop = tf.nn.dropout(L1, keep_prob, name='L1_drop')

    with tf.name_scope('L2'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([300])+0.1, name='b2')
        with tf.name_scope('tanh'):
            L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2, name='tanh')
        with tf.name_scope('L2_drop'):
            L2_drop = tf.nn.dropout(L2, keep_prob, name='L2_drop')

    with tf.name_scope('L3'):
        with tf.name_scope('W3'):
            W3 = tf.Variable(tf.truncated_normal([300, 13], stddev=0.1), name='W3')
        with tf.name_scope('b3'):
            b3 = tf.Variable(tf.zeros([13])+0.1, name='b3')
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(tf.matmul(L2_drop, W3)+b3)
        return prediction