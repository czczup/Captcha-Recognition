import tensorflow as tf

class Model(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, 28, 40, 1], name='X')
            self.Y = tf.placeholder(tf.float32, [None], name='Y')
            label = tf.one_hot(indices=tf.cast(self.Y, tf.int32), depth=31, name='y_onehot')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.prediction, self.parameters = self.model()

        with tf.name_scope('output'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(label, 1))
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=label))
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('AdamOptimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def model(self, w_alpha=0.01, b_alpha=0.1):
        with tf.name_scope('CONV1'):
            w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 64]), name='w_c1')
            b_c1 = tf.Variable(b_alpha*tf.random_normal([64]), name='b_c1')
            conv1 = tf.nn.conv2d(self.X, w_c1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
            wx_plus_b_1 = tf.nn.bias_add(conv1, b_c1, name='wx_plus_b')
            relu_c1 = tf.nn.relu(wx_plus_b_1, name='relu')
            pool1 = tf.nn.max_pool(relu_c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            drop_c1 = tf.nn.dropout(pool1, self.keep_prob, name='drop_c1')
        # 14*20
        with tf.name_scope('CONV2'):
            w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 128]), name='w_c2')
            b_c2 = tf.Variable(b_alpha*tf.random_normal([128]), name='b_c2')
            conv2 = tf.nn.conv2d(drop_c1, w_c2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
            wx_plus_b_2 = tf.nn.bias_add(conv2, b_c2, name='wx_plus_b')
            relu_c2 = tf.nn.relu(wx_plus_b_2, name='relu')
            pool2 = tf.nn.max_pool(relu_c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            drop_c2 = tf.nn.dropout(pool2, self.keep_prob, name='drop_c2')
        # 7*10
        with tf.name_scope('CONV3'):
            w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 256]), name='w_c3')
            b_c3 = tf.Variable(b_alpha*tf.random_normal([256]), name='b_c3')
            conv3 = tf.nn.conv2d(drop_c2, w_c3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
            wx_plus_b_3 = tf.nn.bias_add(conv3, b_c3, name='wx_plus_b')
            relu_c3 = tf.nn.relu(wx_plus_b_3, name='relu')
            pool3 = tf.nn.max_pool(relu_c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            drop_c3 = tf.nn.dropout(pool3, self.keep_prob, name='drop_c3')
        # 4*5
        with tf.name_scope('CONV4'):
            w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 256, 512]), name='w_c4')
            b_c4 = tf.Variable(b_alpha*tf.random_normal([512]), name='b_c4')
            conv4 = tf.nn.conv2d(drop_c3, w_c4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
            wx_plus_b_4 = tf.nn.bias_add(conv4, b_c4, name='wx_plus_b')
            relu_c4 = tf.nn.relu(wx_plus_b_4, name='relu')
            pool4 = tf.nn.max_pool(relu_c4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
            drop_c4 = tf.nn.dropout(pool4, self.keep_prob, name='drop_c4')
        # 2*3
        with tf.name_scope('FC1'):
            dense = tf.reshape(drop_c4, [-1, 2*3*512], name='dense')
            w_fc1 = tf.Variable(w_alpha*tf.random_normal([2*3*512, 1024]), name='w_fc1')
            b_fc1 = tf.Variable(b_alpha*tf.random_normal([1024]), name='b_fc1')
            relu_fc1 = tf.nn.relu(tf.add(tf.matmul(dense, w_fc1), b_fc1), name='relu')
            drop_fc1 = tf.nn.dropout(relu_fc1, self.keep_prob)

        with tf.name_scope('FC2'):
            w_fc2 = tf.Variable(w_alpha*tf.random_normal([1024, 31]), name='w_fc2')
            b_fc2 = tf.Variable(b_alpha*tf.random_normal([31]), name='b_fc2')
            wx_plus_b_fc2 = tf.add(tf.matmul(drop_fc1, w_fc2), b_fc2, name='wx_plus_b')

        return wx_plus_b_fc2, [w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_c4, b_c4]

if __name__ == '__main__':
    model = Model()


