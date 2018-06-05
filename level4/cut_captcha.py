import cv2
import numpy as np
from util import read_CSV
import conf

def cut_batch(start,end):
    """ Cut captcha into 4 parts. """
    num2idx = read_CSV(conf.TRAIN_MAPPINGS)
    for i in range(start,end):
        name = str(i).zfill(4)
        path = conf.TRAIN_IMAGE_PATH+"/"+name+".jpg"
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        # The size of images is (36,36).
        cut_list = [image[4:40,2:38],image[4:40,38:74],image[4:40,74:110],image[4:40,110:146]]

        for j in range(len(cut_list)):
            if j == int(num2idx[i]):
                path = conf.CUT_PATH+"/0/"+name+"_"+str(j)+".png" # Generate save path.
            else:
                path = conf.CUT_PATH+"/1/"+name+"_"+str(j)+".png"
            cv2.imwrite(path,cut_list[j])

        print("Cutting pictures:" + str(i) + "/" + str(end-1))

if __name__ == '__main__':
    cut_batch(0,9500)
