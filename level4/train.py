import tensorflow as tf
from model import Model
import conf
from accuracy_calculate import accuracy_calculate
import time


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
    print(image)
    image = tf.reshape(image, [36, 36, 1])
    image = tf.cast(image, tf.float32)/255.0
    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    # label = tf.reshape(label, [1])
    return image, label


def load_dataset():
    # Load training set.
    with tf.name_scope('input_train'):
        image_train, label_train = read_and_decode("tfrecord/image_train.tfrecord")
        image_batch_train, label_batch_train = tf.train.shuffle_batch(
            [image_train, label_train], batch_size=256, capacity=25600, min_after_dequeue=12800
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
    model = Model()
    sess = tf.Session()

    # Recording training process.
    writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
    writer_valid = tf.summary.FileWriter('logs/valid/', sess.graph)

    # Load training set and validation set.
    image_batch_train, label_batch_train, image_batch_valid, label_batch_valid = load_dataset()

    # General setting.
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    i = 0
    while 1:
        # Get a batch of training set.
        batch_x_train, batch_y_train = sess.run([image_batch_train, label_batch_train])

        # train
        _, loss_value = sess.run([model.optimizer, model.loss], feed_dict={model.X: batch_x_train,
                                                                           model.Y: batch_y_train,
                                                                           model.keep_prob: 0.8})
        print("step "+str(i)+",loss "+str(loss_value))

        if i % 10 == 0:
            # Calculate the accuracy of training set.
            acc_train, summary = sess.run([model.accuracy, model.merged], feed_dict={model.X: batch_x_train,
                                                                                     model.Y: batch_y_train,
                                                                                     model.keep_prob: 1.0})
            writer_train.add_summary(summary, i)

            # Get a batch of validation set.
            batch_x_valid, batch_y_valid = sess.run([image_batch_valid, label_batch_valid])

            # Calculate the accuracy of validation set.
            acc_valid, summary = sess.run([model.accuracy, model.merged], feed_dict={model.X: batch_x_valid,
                                                                                     model.Y: batch_y_valid,
                                                                                     model.keep_prob: 1.0})
            writer_valid.add_summary(summary, i)
            print("step {},Train Accuracy {:.4f},Valid Accuracy {:.4f}".format(i, acc_train, acc_valid))

        if i % 100 == 0:
            print("Save the model Successfully")
            saver.save(sess, "model/model_level4.ckpt", global_step=i)

        i += 1

    coord.request_stop()
    coord.join(threads)


if __name__=='__main__':
    train()
