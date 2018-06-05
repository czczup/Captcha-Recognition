from model import Model
import tensorflow as tf
import numpy as np
import cv2
import conf
import sys

def test():
    model = Model()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Restore the model.
    saver.restore(sess, conf.MODEL_PATH)

    f = open(conf.MAPPINGS, "w")

    for i in range(conf.TEST_NUMBER):
        # Open images.
        name = str(i).zfill(4)
        path = conf.TEST_IMAGE_PATH+"/"+name+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Cut the captcha into four characters.
        cut_list = [image[2:38, 4:40], image[2:38, 40:76],image[2:38, 76:112], image[2:38, 112:148]]
        cut_list = [image.reshape([36,36,1])/255.0 for image in cut_list]

        # Get predictions.
        prediction_list = sess.run(model.prediction, feed_dict={model.X: cut_list, model.keep_prob: 1.0})
        prediction_list = [item[0] for item in prediction_list] # Get the first column.

        # Write predictions into mappings.txt.
        f.write(str(i).zfill(4)+",")
        max_index = prediction_list.index(max(prediction_list))
        f.write(str(max_index)+'\n')

        sys.stdout.write('\r>> Testing image %d/%d' % (i+1,conf.TEST_NUMBER))
        sys.stdout.flush()

if __name__ == '__main__':
    test()