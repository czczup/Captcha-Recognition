import sys
import cv2
import numpy as np
import tensorflow as tf
import conf
import cut_captcha
from model import x, keep_prob, captcha_nn
import denoise_opencv

num2str = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
           7: "7", 8: "8", 9: "9", 10: "*", 11: "+", 12: "-"}


def open_image(i):
    """ Open a RGB image and return a GRAY image. """
    name = str(i).zfill(4)
    # Open a image.
    image = cv2.imread(conf.TEST_IMAGE_PATH+"/"+name+".jpg")
    # Remove the noise on image.
    image = denoise_opencv.RGB_clean(image)
    # Gray processing.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, image_binary = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    # Remove the noise on image.
    image_binary = denoise_opencv.pixel_clean(image_binary)
    # Repair the image.
    image_binary = denoise_opencv.pixel_repair(image_binary)

    return image_binary


def cut(image):
    """ Find the characters and cut them into a list. """
    horizontal_sum = np.sum(image, axis=0)
    cut_list = cut_captcha.image_cut(horizontal_sum, image)
    return cut_list


def test():
    # Network
    prediction = captcha_nn()
    result = tf.argmax(prediction, 1)

    # General setting.
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Restore the model.
    saver.restore(sess, conf.MODEL_PATH)

    f = open(conf.MAPPINGS, "w")

    for i in range(conf.TEST_NUMBER):
        # Open images
        image_binary = open_image(i)  # Get a gray image.
        image_list = cut(image_binary)  # cut

        # Get predictions
        expression = []
        for image in image_list:
            image = np.reshape(image, [-1, 2000])/255.0
            key = sess.run(result, feed_dict={x: image, keep_prob: 1.0})[0]
            expression.append(num2str[key])

        predict_result = "".join(expression)

        # Write predictions into file.
        f.write(str(i).zfill(4)+",")
        f.write(predict_result)
        f.write("="+str(eval(predict_result))+"\n")

        sys.stdout.write('\r>> Testing image %d/%d'%(i+1, conf.TEST_NUMBER))
        sys.stdout.flush()


if __name__=='__main__':
    test()
