# -*- coding:utf-8 -*-

import numpy as np
import pic_model
import os
import configs
import utils
from PIL import Image


def gen_pics(size, mode='test'):
    if mode == 'test':
        path = configs.DATA_SET_PATH + '/test/'
    elif mode == 'train':
        path = configs.DATA_SET_PATH + '/train/'

    if not os.path.exists(path):
        os.makedirs(path)

    ft = pic_model.pic_model()

    for i in range(size):
        length = np.random.randint(1, configs.MAX_LEN + 1)
        label = ""
        for j in range(length):
            idx = np.random.randint(0, len(configs.CHARS_SET))
            label += configs.CHARS_SET[idx]

        filename = str(length) + '_' + label + '_' + utils.gen_eight_bit()

        while os.path.exists(path + filename + '.png'):
            filename = str(length) + '_' + label + '_' + utils.gen_eight_bit()

        image = ft.draw_text(utils.gen_background_image(), configs.TEXT_POS, label, \
                             configs.FONT_SIZE, utils.gen_random_color())
        image = Image.fromarray(image)
        image.save(path + filename + '.png')


if __name__ == '__main__':
    if not os.path.exists(configs.DATA_SET_PATH):
        os.makedirs(configs.DATA_SET_PATH)
    print("Start Generating %d Training Pics..." % configs.NUM_PICS_TRAIN)
    gen_pics(configs.NUM_PICS_TRAIN, mode='train')
    print("Start Generating %d Test Pics..." % configs.NUM_PICS_TEST)
    gen_pics(configs.NUM_PICS_TEST, mode='test')
    print("Data Set Generation Finished!")
