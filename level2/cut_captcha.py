import cv2
import denoise_opencv
from multiprocessing import Process
from util import read_CSV
import conf

def cut_batch(start,end):
    num2char = read_CSV(conf.TRAIN_MAPPINGS)
    for i in range(start,end):
        name = str(i).zfill(4)
        path = conf.TRAIN_IMAGE_PATH+"/"+name+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cnt = 0
        cut_list = [image[:,2:46],image[:,37:81],image[:,77:121],image[:,117:161],image[:,154:198]]
        for j in range(len(cut_list)):
            path = conf.CUT_PATH+"/"+num2char[i][j]+"/"+name+"_"+str(j)+"_"+str(cnt)+".png"
            image = denoise_opencv.remove_noise(cut_list[j])
            cv2.imwrite(path,image)
        print(">> Cutting pictures:" + str(i+1) + "/" + str(end))

if __name__ == '__main__':
    p1 = Process(target=cut_batch, args=(0, 2000,))
    p2 = Process(target=cut_batch, args=(2000, 4000,))
    p3 = Process(target=cut_batch, args=(4000, 6000,))
    p4 = Process(target=cut_batch, args=(6000, 8000,))
    p5 = Process(target=cut_batch, args=(8000, 10000,))
    pool = [p1, p2, p3, p4, p5]
    for p in pool:
        p.start()
