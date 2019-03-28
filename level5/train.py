from model import Siamese
import tensorflow as tf
import conf
import time
import heapq
import sys

from test import test
from accuracy_calculate import accuracy_calculate


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image1': tf.FixedLenFeature([], tf.string),
                                           'image2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image1 = tf.reshape(image1, [45, 45, 1])
    image1 = tf.random_crop(image1, [40, 40, 1])
    image1 = tf.cast(image1, tf.float32)/255.0

    image2 = tf.decode_raw(features['image2'], tf.uint8)
    image2 = tf.reshape(image2, [45, 45, 1])
    image2 = tf.random_crop(image2, [40, 40, 1])
    image2 = tf.cast(image2, tf.float32)/255.0

    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    label = tf.reshape(label, [1])
    return image1, image2, label


def load_training_set():
    # Load training set.
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode("tfrecord/image_train.tfrecord")
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=10240, min_after_dequeue=2560
        )
    return image_batch_train1, image_batch_train2, label_batch_train


def load_testing_set():
    # Load Testing set.
    with tf.name_scope('input_test'):
        image_test1, image_test2, label_test = read_and_decode("tfrecord/image_valid.tfrecord")
        image_batch_test1, image_batch_test2, label_batch_test = tf.train.shuffle_batch(
            [image_test1, image_test2, label_test], batch_size=256, capacity=1280, min_after_dequeue=512
        )
    return image_batch_test1, image_batch_test2, label_batch_test


def train():
    # network
    siamese = Siamese()

    image_batch_train1, image_batch_train2, label_batch_train = load_training_set()
    image_batch_test1, image_batch_test2, label_batch_test = load_testing_set()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # general setting
        saver = tf.train.Saver(max_to_keep=20)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, conf.MODEL_PATH)

        # Recording training process.
        writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
        writer_test = tf.summary.FileWriter('logs/test/', sess.graph)
        saver = tf.train.Saver(max_to_keep=20)
        # train
        i = 0
        acc_max = 0
        while 1:
            image_train1, image_train2, label_train = sess.run(
                [image_batch_train1, image_batch_train2, label_batch_train])
            _, loss_ = sess.run([siamese.optimizer, siamese.loss], feed_dict={siamese.left: image_train1,
                                                                              siamese.right: image_train2,
                                                                              siamese.label: label_train})
            print('step %d: loss %.3f'%(i, loss_))

            if i%10==0:
                image_train1, image_train2, label_train = sess.run(
                    [image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_train1,
                                                                                             siamese.right: image_train2,
                                                                                             siamese.label: label_train})
                writer_train.add_summary(summary, i)
                image_test1, image_test2, label_test = sess.run(
                    [image_batch_test1, image_batch_test2, label_batch_test])
                acc_test, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_test1,
                                                                                            siamese.right: image_test2,
                                                                                            siamese.label: label_test})
                writer_test.add_summary(summary, i)
                print("Lter "+str(i)+",Train Accuracy "+str(acc_train)+",Test Accuracy "+str(acc_test))

            if i%100==0:
                test(siamese, sess)
                acc = accuracy_calculate()
                if acc>acc_max:
                    acc_max = acc
                    print("Save the model Successfully,max accuracy is", acc_max)
                    saver.save(sess, "model/model_level5.ckpt", global_step=i)
                else:
                    print("pass,max accuracy is", acc_max)
            i += 1

    coord.request_stop()
    coord.join(threads)


if __name__=='__main__':
    train()
