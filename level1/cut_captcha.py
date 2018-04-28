import cv2
import numpy as np
import conf
from util import read_CSV
from multiprocessing import Process

name_dic = {"1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7",
            "8":"8","9":"9","0":"0","+":"plus","-":"sub","*":"mul"}

def image_cut(sum,image,threshold=19600):
    """ Find the characters and cut them into a single file. """
    side1_available, side2_available = True,True
    width = 40
    cut_list = []
    for i in range(image.shape[1]):
        if sum[i] < threshold and side1_available:
            side1 = i
            side1_available = False
            continue
        if sum[i] > threshold and side2_available and not side1_available:
            if i - side1 > 30:
                side2 = i
                side2_available = False
            continue
        if not side1_available and  not side2_available:
            distance = abs(side2 - side1)
            if np.sum(sum[side1:side2]) / distance < 20000: # 切片平均累加灰度值
                if distance > width:
                    side1 += (distance - width) // 2
                    side2 -= ((distance - width) - (distance - width)//2)
                elif distance < width:
                    side1 -= (width - distance)//2
                    side2 += ((width - distance) - (width - distance)//2)
                cut_list.append(image[19:69,side1:side2]) #切割图片
            side1_available,side2_available = True,True
    return cut_list

def cut_batch(start,end):
    """ Batch processing. """
    mappings = read_CSV("raw_mappings.txt")
    for i in range(start,end):
        name = str(i).zfill(4)
        image = cv2.imread(conf.DENOISE_PATH+"/"+name+".png")
        horizontal_sum = np.sum(image, axis=0)
        sum = [item[0] for item in horizontal_sum]
        cut_list = image_cut(sum,image)
        for j in range(len(cut_list)):
            path = conf.CUT_PATH+"/"+name_dic[mappings[i][j]]+"/"+name+"_"+str(j)+".png"
            cv2.imwrite(path, cut_list[j])
        print("Cutting pictures:" + str(i) + "/" + str(end-1))

if __name__ == '__main__':
    p1 = Process(target=cut_batch, args=(0,2000,))
    p2 = Process(target=cut_batch, args=(2000,4000,))
    p3 = Process(target=cut_batch, args=(4000,6000,))
    p4 = Process(target=cut_batch, args=(6000,8000,))
    p5 = Process(target=cut_batch, args=(8000,9500,))
    pool = [p1, p2, p3, p4, p5]
    for p in pool:
        p.start()

