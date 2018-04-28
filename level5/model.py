import tensorflow as tf


class Siamese(object):
    def __init__(self):
        with tf.name_scope("input"):
            self.left = tf.placeholder(tf.float32, [None, 40, 40, 1], name='left')
            self.right = tf.placeholder(tf.float32, [None, 40, 40, 1], name='right')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.name_scope("similarity"):
            label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
            self.label = tf.to_float(label)

        self.left_output = self.siamesenet(self.left, reuse=False)
        self.right_output = self.siamesenet(self.right, reuse=True)
        self.y_, self.loss = self.contrastive_loss(self.left_output,self.right_output,self.label)

    def siamesenet(self, input, reuse=False):
        with tf.name_scope("Siamese"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(input, 32, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            # 20*20*32
            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            # 10*10*64
            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            # 5*5*128
            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                output1 = tf.contrib.layers.flatten(net)
            # 5*5*128
            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                output2 = tf.contrib.layers.flatten(net)
            # 3*3*256
            with tf.variable_scope("conv6") as scope:
                net = tf.contrib.layers.conv2d(net, 32, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
                output3 = tf.contrib.layers.flatten(net)
            # 2*2*32
            net = tf.concat([output1, output2, output3], 1)

            # add hidden layer1
            hidden_Weights1 = tf.Variable(tf.truncated_normal([5632, 1024], stddev=0.1)) # 45-7040 40-5632
            hidden_biases1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            net = tf.nn.relu(tf.matmul(net, hidden_Weights1) + hidden_biases1)
            # add hidden layer2
            hidden_Weights2 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1)) # 128
            hidden_biases2 = tf.Variable(tf.constant(0.1, shape=[256]))
            net = tf.nn.relu(tf.matmul(net, hidden_Weights2) + hidden_biases2)
        return net

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            output_difference = tf.abs(model1 - model2)
            W = tf.Variable(tf.random_normal([256, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            y_ = tf.nn.sigmoid(tf.matmul(output_difference, W) + b, name='distance')
        # Calculate mean loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_,labels=y)
            loss = tf.reduce_mean(losses)
        return y_, loss
