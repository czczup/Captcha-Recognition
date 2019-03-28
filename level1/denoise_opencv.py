import numpy as np
import sys
from multiprocessing import Process
import cv2
import conf


def get_threshold(image, x=26, y=64):
    """
    This is a method to compute the threshold of RGB images.
    Threshold is defined as the RGB color in characters.
    In order to simplify this task, we can use location (26,64)
    as sampling point(if the number is in [0,1,2,3,5,6,7,8,9]).
    If the character is "4", try to move up the sampling point. 
    """
    pixels = np.array([image[y-1, x-1], image[y-1, x], image[y-1, x+1],
                       image[y, x-1], image[y, x], image[y, x+1],
                       image[y+1, x-1], image[y+1, x], image[y+1, x+1]])
    mean = np.sum(pixels, axis=0)//9
    # If average of mean is less than 160, that means the threshold is valid.
    # Else the threshold is invalid, try to move up the sampling point.
    if sum(mean)//3 < 160:
        return mean
    else:
        return get_threshold(image, x, y-3)


def RGB_clean(image):
    """ Remove noisy points according to RGB thresholds. """
    B, G, R = [int(i) for i in get_threshold(image)]
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            b, g, r = [int(i) for i in image[y, x]]
            if abs(b-B) > 30 or abs(g-G)>30 or abs(r-R)>30:
                image[y, x] = (255, 255, 255)
    return image


def pixel_repair(image):
    """ Repair the holes in characters caused by removing noisy points. """
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            num = sum_9_region(image, x, y, cal_white=False)
            if num > 4:
                image[y, x] = 0
    return image


def pixel_clean(image):
    """ Remove noisy points according to amount of black points in 9 regions. """
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            num = sum_9_region(image, x, y)
            if num < 5:
                image[y, x] = 255
    return image


def sum_9_region(img, x, y, cal_white=True):
    """ Count the amount of black points in 9 regions. """
    cur_pixel = img[y, x]  # current pixel
    width = img.shape[1]
    height = img.shape[0]
    if cal_white:
        if int(cur_pixel)==255:  # If current pixel is white, return 0 directly.
            return 0
    if y == 0:  # first column
        if x == 0:
            pixels = [cur_pixel, img[y+1, x], img[y, x+1], img[y+1, x+1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 4-sum_value//255
        elif x == width-1:
            pixels = [cur_pixel, img[y+1, x], img[y, x-1], img[y+1, x-1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 4-sum_value//255
        else:
            pixels = [cur_pixel, img[y, x-1], img[y+1, x-1], img[y+1, x], img[y, x+1], img[y+1, x+1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 6-sum_value//255
    elif y == height-1:  # last column
        if x == 0:
            pixels = [cur_pixel, img[y, x+1], img[y-1, x+1], img[y-1, x]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 4-sum_value//255
        elif x == width-1:
            pixels = [cur_pixel, img[y-1, x], img[y, x-1], img[y-1, x-1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 4-sum_value//255
        else:
            pixels = [cur_pixel, img[y, x-1], img[y, x+1], img[y-1, x], img[y-1, x-1], img[y-1, x+1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 6-sum_value//255
    else:  # middle columns
        if x == 0:
            pixels = [cur_pixel, img[y-1, x], img[y+1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 6-sum_value//255
        elif x == width-1:
            pixels = [cur_pixel, img[y-1, x], img[y+1, x], img[y-1, x-1], img[y, x-1], img[y+1, x-1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 6-sum_value//255
        else:
            pixels = [img[y-1, x-1], img[y, x-1], img[y+1, x-1], img[y-1, x], cur_pixel,
                      img[y+1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1]]
            sum_value = sum([int(pixel) for pixel in pixels])
            return 9-sum_value//255


def remove_noise(num):
    """ A complete noise points reduction process."""
    name = str(num).zfill(4)
    # Open a image.
    image = cv2.imread(conf.TRAIN_IMAGE_PATH+"/"+name+".jpg")
    # Remove the noise on image.
    image = RGB_clean(image)
    # Gray processing.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, image_binary = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    # Remove the noise on image.
    image_binary = pixel_clean(image_binary)
    # Repair the image.
    image_binary = pixel_repair(image_binary)
    # Save the image.
    cv2.imwrite(conf.DENOISE_PATH+"/"+name+".png", image_binary)


def remove_batch(start, end):
    """ Batch processing. """
    for i in range(start, end):
        remove_noise(i)
        print("Dealing pictures:"+str(i)+"/"+str(end-1))


if __name__=='__main__':
    p1 = Process(target=remove_batch, args=(0, 2000,))
    p2 = Process(target=remove_batch, args=(2000, 4000,))
    p3 = Process(target=remove_batch, args=(4000, 6000,))
    p4 = Process(target=remove_batch, args=(6000, 8000,))
    p5 = Process(target=remove_batch, args=(8000, 9500,))
    pool = [p1, p2, p3, p4, p5]
    for p in pool:
        p.start()
