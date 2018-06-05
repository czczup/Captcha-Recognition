import tensorflow as tf
from model import Siamese
import numpy as np
import cv2
import conf
import sys
import heapq
import time

def test(siamese, sess):
    f = open(conf.MAPPINGS, "w")
    time1 = time.time()
    for i in range(conf.TEST_NUMBER):
        name = str(i).zfill(4)
        path = conf.TEST_IMAGE_PATH+"/"+name
        image = cv2.imread(path+"/"+name+".jpg", cv2.IMREAD_GRAYSCALE)
        cut_list = [image[3:43, 2:42], image[2:42, 34:74],image[2:42, 75:115], image[3:43, 108:148]]
        # cut_list = [image[3:43, 3:43], image[3:43, 39:79], image[3:43, 73:113], image[3:43, 107:147]]
        result = []
        for j in range(4):
            lst1, lst2 = [],[]
            image1 = cut_list[j].reshape([40,40,1]) / 255.0
            for k in range(9):
                image2 = cv2.imread(path+"/"+str(k)+".jpg", cv2.IMREAD_GRAYSCALE)
                image2 = image2[3:43,3:43].reshape([40,40,1]) / 255.0
                lst1.append(image1)
                lst2.append(image2)

            prediction = sess.run(siamese.prediction,feed_dict={siamese.left:lst1,siamese.right:lst2})
            max_list = heapq.nlargest(4, range(len(prediction)), prediction.take)
            for num in max_list:
                if num not in result:
                    result.append(num)
                    break

        # Write predictions into mappings.txt.
        f.write(str(i).zfill(4)+",")
        f.write("".join([str(num) for num in result])+'\n')
        sys.stdout.write('\r>> Testing image %d/%d'%(i+1, conf.TEST_NUMBER))
        sys.stdout.flush()
    time2 = time.time()
    print("\nUsing time:","%.2f"%(time2-time1)+"s")


if __name__ == '__main__':
    # Network
    siamese = Siamese()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    # Restore the model.
    saver = tf.train.Saver()
    saver.restore(sess, conf.MODEL_PATH)

    test(siamese, sess)