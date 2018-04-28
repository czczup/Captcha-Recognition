import cv2
from util import read_CSV
import conf

num2char = read_CSV("raw_mappings.txt")

def cut_batch(start,end):
    for i in range(start,end):
        name = str(i).zfill(4)
        path = conf.TRAIN_IMAGE_PATH+"/"+name+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cnt = 6
        cut_list = [image[:,6:62],image[:,53:109],image[:,96:152],image[:,138:194]]
        for j in range(4):
            path = conf.CUT_PATH+"/"+num2char[i][j]+"/"+name+"_"+str(j)+"_"+str(cnt)+".png"
            if cut_list[j].shape == (80,56):
                _, image = cv2.threshold(cut_list[j], 180, 255, cv2.THRESH_BINARY)
                cv2.imwrite(path,image)
        print("Cutting pictures:"+str(i)+"/"+str(end-1))

if __name__ == '__main__':
    cut_batch(0,9500)