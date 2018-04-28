from model import *
from util import read_CSV
import numpy as np
import conf
import time
import cv2
import heapq
import sys
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename]) # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image1':tf.FixedLenFeature([],tf.string),
                                           'image2': tf.FixedLenFeature([],tf.string),
                                           'label':tf.FixedLenFeature([],tf.int64),
                                       }) # return image and label
    image1 = tf.decode_raw(features['image1'],tf.uint8)
    image1 = tf.reshape(image1,[45,45,1])
    image1 = tf.random_crop(image1,[40,40,1])
    image1 = tf.cast(image1, tf.float32) / 255.0

    image2 = tf.decode_raw(features['image2'], tf.uint8)
    image2 = tf.reshape(image2, [45,45,1])
    image2 = tf.random_crop(image2, [40, 40, 1])
    image2 = tf.cast(image2, tf.float32) / 255.0

    label = tf.cast(features['label'],tf.int64) # throw label tensor
    label = tf.reshape(label,[1])
    return image1, image2, label

def load_training_set():
    # Load training set.
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode("tfrecord/image_train.tfrecord")
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=10240, min_after_dequeue=2560
        )
    return image_batch_train1, image_batch_train2, label_batch_train

def load_validation_set():
    # Load validation set.
    with tf.name_scope('input_valid'):
        image_valid1, image_valid2, label_valid = read_and_decode("tfrecord/image_valid.tfrecord")
        image_batch_valid1, image_batch_valid2, label_batch_valid = tf.train.shuffle_batch(
            [image_valid1, image_valid2, label_valid], batch_size=256, capacity=1280, min_after_dequeue=512
        )
    return image_batch_valid1, image_batch_valid2, label_batch_valid

def train():
    # network
    siamese = Siamese()
    correct_prediction = tf.equal(tf.less(siamese.y_, 0.5),tf.less(siamese.label,0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(siamese.loss)
    tf.summary.scalar('loss', siamese.loss)
    merged = tf.summary.merge_all()

    image_batch_train1, image_batch_train2, label_batch_train = load_training_set()
    image_batch_valid1, image_batch_valid2, label_batch_valid = load_validation_set()

    with tf.Session() as sess:
        # general setting
        # var = tf.global_variables()[0:20]
        # saver = tf.train.Saver(var,max_to_keep=20)
        saver = tf.train.Saver(max_to_keep=20)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,conf.MODEL_PATH)
        # Recording training process.
        writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
        writer_valid = tf.summary.FileWriter('logs/valid/', sess.graph)
        # saver = tf.train.Saver(max_to_keep=20)

        # train
        i = 0
        while 1:
            image_train1,image_train2,label_train = sess.run([image_batch_train1, image_batch_train2, label_batch_train])
            _, loss_ = sess.run([optimizer, siamese.loss],feed_dict={siamese.left: image_train1,
                                                                     siamese.right: image_train2,
                                                                     siamese.label: label_train})
            print('step %d: loss %.3f'%(i, loss_))

            if i%10 == 0:
                image_train1, image_train2, label_train = sess.run([image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([accuracy, merged],feed_dict={siamese.left: image_train1,
                                                                            siamese.right: image_train2,
                                                                            siamese.label: label_train})
                writer_train.add_summary(summary, i)
                image_valid1, image_valid2, label_valid = sess.run([image_batch_valid1, image_batch_valid2, label_batch_valid])
                acc_valid = sess.run(accuracy, feed_dict={siamese.left: image_valid1,
                                                         siamese.right: image_valid2,
                                                         siamese.label: label_valid})
                writer_valid.add_summary(summary, i)
                print("Lter "+str(i)+",Train Accuracy "+str(acc_train)+",Valid Accuracy "+str(acc_valid))

            if i % 100 == 0 and i != 0:
                print("Save the model Successfully")
                saver.save(sess, "model/model_level5.ckpt", global_step=i)
            i += 1

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    train()
