import tensorflow as tf

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 36*36], name='X')
    x = tf.reshape(X, shape=[-1, 36, 36, 1])
    Y = tf.placeholder(tf.float32, [None], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    with tf.name_scope('CONV1'):
        w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]),name='w_c1')
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]),name='b_c1')
        conv1 = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME',name='conv1')
        wx_plus_b_1 = tf.nn.bias_add(conv1,b_c1,name='wx_plus_b')
        relu_c1 = tf.nn.relu(wx_plus_b_1,name='relu')
        pool1 = tf.nn.max_pool(relu_c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool1')
        drop_c1 = tf.nn.dropout(pool1, keep_prob,name='drop_c1')

    with tf.name_scope('CONV2'):
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 32]),name='w_c2')
        b_c2 = tf.Variable(b_alpha * tf.random_normal([32]),name='b_c2')
        conv2 = tf.nn.conv2d(drop_c1, w_c2, strides=[1, 1, 1, 1], padding='SAME',name='conv2')
        wx_plus_b_2 = tf.nn.bias_add(conv2, b_c2,name='wx_plus_b')
        relu_c2 = tf.nn.relu(wx_plus_b_2,name='relu')
        pool2 = tf.nn.max_pool(relu_c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool2')
        drop_c2 = tf.nn.dropout(pool2, keep_prob,name='drop_c2')

    with tf.name_scope('CONV3'):
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]),name='w_c3')
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]),name='b_c3')
        conv3 = tf.nn.conv2d(drop_c2, w_c3, strides=[1, 1, 1, 1], padding='SAME',name='conv3')
        wx_plus_b_3 = tf.nn.bias_add(conv3, b_c3,name='wx_plus_b')
        relu_c3 = tf.nn.relu(wx_plus_b_3,name='relu')
        pool3 = tf.nn.max_pool(relu_c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool3')
        drop_c3 = tf.nn.dropout(pool3, keep_prob,name='drop_c3')

    with tf.name_scope('FC1'):
        dense = tf.reshape(drop_c3, [-1, 5*5*64],name='dense')
        w_fc1 = tf.Variable(w_alpha * tf.random_normal([5*5*64, 512]),name='w_fc1')
        b_fc1 = tf.Variable(b_alpha * tf.random_normal([512]),name='b_c3')
        relu_fc1 = tf.nn.relu(tf.add(tf.matmul(dense, w_fc1), b_fc1),name='relu')
        drop_fc1 = tf.nn.dropout(relu_fc1, keep_prob)

    with tf.name_scope('FC2'):
        w_fc2 = tf.Variable(w_alpha * tf.random_normal([512, 2]),name='w_fc2')
        b_fc2 = tf.Variable(b_alpha * tf.random_normal([2]),name='b_fc2')
        wx_plus_b_fc2 = tf.add(tf.matmul(drop_fc1, w_fc2), b_fc2,name='wx_plus_b')
        return wx_plus_b_fc2

