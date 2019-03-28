import tensorflow as tf
from model import lr, x, y, Y, keep_prob, captcha_nn
import conf


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [2000])  # reshape image to 40*50
    image = tf.cast(image, tf.float32)/255.0
    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    return image, label


def network():
    prediction = captcha_nn()
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('AdamOptimizer'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))
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
            [image_train, label_train], batch_size=128, capacity=12800, min_after_dequeue=5120
        )
    # Load validation set.
    with tf.name_scope('input_valid'):
        image_valid, label_valid = read_and_decode("tfrecord/image_valid.tfrecord")
        image_batch_valid, label_batch_valid = tf.train.shuffle_batch(
            [image_valid, label_valid], batch_size=256, capacity=12800, min_after_dequeue=5120
        )
    return image_batch_train, label_batch_train, image_batch_valid, label_batch_valid


def train():
    # Network
    loss, optimizer, accuracy, merged = network()
    sess = tf.Session()

    # Recording training process.
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    valid_writer = tf.summary.FileWriter('logs/valid', sess.graph)

    # Load training set and validation set.
    image_batch_train, label_batch_train, image_batch_valid, label_batch_valid = load_dataset()

    # General setting.
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    for i in range(300):
        # Learning rate delay.
        if i % 40 == 0:
            sess.run(tf.assign(lr, 0.001*(0.95**(i//40))))

        # Get a batch of data.
        batch_x_train, batch_y_train = sess.run([image_batch_train, label_batch_train])
        batch_x_valid, batch_y_valid = sess.run([image_batch_valid, label_batch_valid])

        # train
        sess.run(optimizer, feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 0.7})

        # Record summary.
        summary = sess.run(merged, feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        summary = sess.run(merged, feed_dict={x: batch_x_valid, y: batch_y_valid, keep_prob: 1.0})
        valid_writer.add_summary(summary, i)

        # Calculate the accuracy of training set and validation set.
        train_acc = sess.run(accuracy, feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
        valid_acc = sess.run(accuracy, feed_dict={x: batch_x_valid, y: batch_y_valid, keep_prob: 1.0})
        print("Lter "+str(i)+",Train Accuracy "+str(train_acc)+",Valid Accuracy "+str(valid_acc))

        # Save the model.
        if valid_acc == 1.00:
            print("Save the model Successfully")
            saver.save(sess, "model/model_level1.ckpt", global_step=i)

    coord.request_stop()
    coord.join(threads)


if __name__=='__main__':
    train()
