import conf
import random
import cv2
import tensorflow as tf
from util import read_CSV
import numpy as np
import conf
# sess = tf.InteractiveSession()
num2idx = read_CSV("raw_mappings.txt")

def get_positive_and_negative_batch():
    for i in range(300000):
        image1, image2, label = get_positive_pair()
        image_1 = np.concatenate([image1, image2], axis=1)
        save_path = conf.CUT_PATH+"/1/"+str(i)+".jpg"
        cv2.imwrite(save_path, image_1)
        image1, image2, label = get_negative_pair()
        image_2 = np.concatenate([image1, image2], axis=1)
        save_path = conf.CUT_PATH+"/0/"+str(i)+".jpg"
        cv2.imwrite(save_path, image_2)
        print("Cutting pictures:"+str(i+1)+"/"+str(300000))


def get_positive_pair(): # 获取正样本对
    num = np.random.randint(0,9500)
    folder = str(num).zfill(4) # [0,9499]
    mode = np.random.randint(1,4) # [1,2,3]
    label = [1]
    if mode == 1: # 从XXXX.jpg中选择
        path = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+folder+".jpg"
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        choose = np.random.randint(0,4) # [0,1,2,3]
        cut_side = [[0,45],[33,78],[72,117],[105,150]]
        left, right = cut_side[choose]
        image1 = image[:,left:right]
        image2 = image[:, left:right]
        return image1,image2,label
    elif mode == 2: # 从XXXX.jpg中选择一个，再从剩下小图片中选择一个
        path = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+folder+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        choose = np.random.randint(0,4) # [0,1,2,3]
        cut_side = [[0, 45], [33, 78], [72, 117], [105, 150]]
        left, right = cut_side[choose]
        image1 = image[:, left:right]
        path2 = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+num2idx[num][choose]+".jpg"
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        return image1, image2, label
    elif mode == 3: # 从小图片中选择一个
        choose = np.random.randint(0, 9) # [0,1,2,3,4,5,6,7,8]
        path = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+str(choose)+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image1 = image
        image2 = image
        return image1, image2, label

def get_negative_pair(): # 获取负样本对
    num = np.random.randint(0, 9500)
    folder = str(num).zfill(4) # [0,9499]
    # mode = np.random.randint(1,4) # [1,2,3]
    label = [0]
    mode = 2
    if mode==1: # 从XXXX.jpg中选择
        path = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+folder+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        choose1, choose2 = random.sample(range(0,4),2)
        cut_side = [[0, 45], [33, 78], [72, 117], [105, 150]]
        left, right = cut_side[choose1]
        image1 = image[:, left:right]
        left, right = cut_side[choose2]
        image2 = image[:, left:right]
        return image1, image2, label
    elif mode==2: # 从XXXX.jpg中选择一个，再从剩下小图片中选择一个
        path = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+folder+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        choose1 = np.random.randint(0,4) # [0,1,2,3]
        cut_side = [[0, 45], [33, 78], [72, 117], [105, 150]]
        left, right = cut_side[choose1]
        image1 = image[:, left:right]
        choose2 = random.sample(set(range(0,9))-{int(num2idx[num][choose1])},1)[0] # [0,8]-num2idx[num][choose]
        path2 = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+str(choose2)+".jpg"
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        return image1, image2, label
    elif mode==3: # 从小图片中选择一个
        choose1 = np.random.randint(0,9) # [0,1,2,3,4,5,6,7,8]
        choose2 = random.sample(set(range(0, 9))-{choose1},1)[0] # [0,8]-num2idx[num][choose]
        path1 = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+str(choose1)+".jpg"
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        path2 = conf.TRAIN_IMAGE_PATH+"/"+folder+"/"+str(choose2)+".jpg"
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        return image1, image2, label

if __name__ == '__main__':
    get_positive_and_negative_batch()