import tensorflow as tf

class Model(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, 36, 36, 1], name='X')
            self.Y = tf.placeholder(tf.float32, [None], name='Y')
            label = tf.one_hot(indices=tf.cast(self.Y, tf.int32), depth=2, name='y_onehot')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.name_scope('output'):
            with tf.name_scope('model'):
                self.prediction = self.model()
            with tf.name_scope('correct_prediction'):
                # correct_prediction = tf.equal(tf.less(prediction, 0.5), tf.less(label, 0.5))
                correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(label, 1))
            with tf.name_scope('loss'):
                # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=label))
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=label))
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('AdamOptimizer'):
                self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


    def model(self):
        with tf.name_scope('conv1'): # conv1: 36*36*32 pool: 18*18*32
            conv1 = tf.contrib.layers.conv2d(self.X, 32, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            norm = tf.layers.batch_normalization(conv1)
            pool = tf.contrib.layers.max_pool2d(norm, [2, 2], padding='SAME')
            dropout = tf.nn.dropout(pool,keep_prob=self.keep_prob)

        with tf.name_scope('conv2'): # conv2: 18*18*32 pool: 9*9*32
            conv2 = tf.contrib.layers.conv2d(dropout, 32, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            norm = tf.layers.batch_normalization(conv2)
            pool = tf.contrib.layers.max_pool2d(norm, [2, 2], padding='SAME')
            dropout = tf.nn.dropout(pool, keep_prob=self.keep_prob)

        with tf.name_scope('conv3'): # conv3: 9*9*64 pool: 5*5*64
            conv3 = tf.contrib.layers.conv2d(dropout, 64, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            norm = tf.layers.batch_normalization(conv3)
            pool = tf.contrib.layers.max_pool2d(norm, [2, 2], padding='SAME')
            dropout = tf.nn.dropout(pool, keep_prob=self.keep_prob)

        with tf.name_scope("fc1"):
            dense = tf.reshape(dropout, [-1, 5*5*64])
            w1 = tf.Variable(tf.truncated_normal([5*5*64, 512], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=[512]))
            wx_plus_b = tf.add(tf.matmul(dense, w1), b1)
            relu_fc1 = tf.nn.relu(wx_plus_b)
            drop_fc1 = tf.nn.dropout(relu_fc1, keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            w2 = tf.Variable(tf.truncated_normal([512, 2], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[2]))
            wx_plus_b = tf.add(tf.matmul(drop_fc1, w2), b2)
            output = tf.nn.softmax(wx_plus_b)

        return output

if __name__ == '__main__':
    model = Model()
    print(model)