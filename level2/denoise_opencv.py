import cv2


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


def remove_noise(image):
    """ A complete noise points reduction process."""
    # Thresholding
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # Remove the noise on image.
    image = pixel_clean(image)
    # Repair the image.
    image = pixel_repair(image)
    return image
