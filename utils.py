# -*- coding:utf-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
from PIL import ImageFilter
import configs
import time


def gen_background_image():
    image = ImageCaptcha(width=configs.PIC_WIDTH, height=configs.PIC_HEIGHT)
    background = image.generate(' ')
    background_img = Image.open(background)
    return np.array(background_img)


def gen_random_color():
    while True:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)

        if r <= 220 and g <= 220 and b <= 220:
            break
    ret = (r, g, b)
    return ret


def gen_time_stamp():
    return int(time.time())


def gen_eight_bit():
    ret = ''
    for i in range(8):
        ret += str(np.random.randint(0, 10))
    return ret


def img_to_numpy(img, threshold=115):
    img = img.convert('L')  # 先转成灰度值
    bin_img = img.point(configs.BIN_TABLE, '1')  # 得到二值化后的黑白图片
    # bin_img.show()
    fil_img = bin_img.filter(ImageFilter.MedianFilter(size=9))
    # fil_img.show()
    return np.array(fil_img, dtype=np.int32)


if __name__ == '__main__':
    img = Image.open('./data/train/4_4872_55959018.png')


