import tensorflow as tf
from model import captcha_cnn,X,Y,keep_prob

def read_and_decode(filename):
    """ Decode the TFRecord files. """
    filename_queue = tf.train.string_input_producer([filename]) # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':tf.FixedLenFeature([],tf.string),
                                           'label':tf.FixedLenFeature([],tf.int64),
                                       }) # return image and label
    image = tf.decode_raw(features['image'],tf.uint8)
    image = tf.reshape(image,[1296])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(features['label'],tf.int64) # throw label tensor
    return image, label

def network():
    output = captcha_cnn()
    prediction = tf.nn.softmax(output)
    label = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=2, name='y_onehot')

    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=label))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('AdamOptimizer'):
        lr = tf.Variable(0.001, dtype=tf.float32, name='lr')
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    return loss, optimizer, accuracy, merged

def load_dataset():
    # Load training set.
    with tf.name_scope('input_train'):
        image_train, label_train = read_and_decode("tfrecord/image_train.tfrecord")
        image_batch_train, label_batch_train = tf.train.shuffle_batch(
            [image_train, label_train], batch_size=128, capacity=51200, min_after_dequeue=12800
        )
    # Load validation set.
    with tf.name_scope('input_valid'):
        image_valid, label_valid = read_and_decode("tfrecord/image_valid.tfrecord")
        image_batch_valid, label_batch_valid = tf.train.shuffle_batch(
            [image_valid, label_valid], batch_size=1024, capacity=4096, min_after_dequeue=2048
        )
    return image_batch_train, label_batch_train, image_batch_valid, label_batch_valid

def train():
    # Network
    loss, optimizer, accuracy, merged = network()
    sess = tf.Session()

    # Recording training process.
    writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
    writer_valid = tf.summary.FileWriter('logs/valid/', sess.graph)

    # Load training set and validation set.
    image_batch_train, label_batch_train, image_batch_valid, label_batch_valid   = load_dataset()

    # general setting
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    i = 0
    while 1:
        # Get a batch of training set.
        batch_x_train, batch_y_train = sess.run([image_batch_train, label_batch_train])

        # training
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x_train, Y: batch_y_train, keep_prob: 0.7})
        print(i, 'loss:\t', loss_)

        # Learning rate delay.
        if i > 1000 and i%1000==0:
            sess.run(tf.assign(lr, 0.001*(0.95**(i//1000))))

        if i%10 == 0:
            # Calculate the accuracy of training set.
            acc_train,summary = sess.run([accuracy, merged],feed_dict={X: batch_x_train, Y: batch_y_train, keep_prob: 1.0})
            writer_train.add_summary(summary, i)
            # Get a batch of validation set.
            batch_x_valid, batch_y_valid = sess.run([image_batch_valid, label_batch_valid])
            # Calculate the accuracy of validation set.
            acc_valid, summary = sess.run([accuracy, merged],feed_dict={X: batch_x_valid, Y: batch_y_valid, keep_prob: 1.0})
            writer_valid.add_summary(summary, i)
            print("Lter "+str(i)+",Train Accuracy "+str(acc_train)+",Valid Accuracy "+str(acc_valid))
            # Save the model.
            if acc_train > 0.99:
                print("Save the model Successfully")
                saver.save(sess, "model/model_level4.ckpt", global_step=i)
        i += 1

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    train()