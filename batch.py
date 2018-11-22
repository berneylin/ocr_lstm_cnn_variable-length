# -*- coding:utf-8 -*-
import numpy as np
import utils
import os
import configs
from PIL import Image
import vec_to_sparse_mat as vtm


def get_train_set():
    train_path = configs.DATA_SET_PATH + '/train/'
    train_pics = os.listdir(train_path)
    np.random.shuffle(train_pics)
    return train_pics


def get_next_batch(train_pics, iter, batch_size=configs.BATCH_SIZE):
    begin_idx = iter * batch_size
    if begin_idx >= len(train_pics):
        return None, None

    ret_x, label_list = [], []

    while begin_idx < len(train_pics) and len(ret_x) < batch_size:
        this_img_filename = train_pics[begin_idx]
        this_img_path = configs.DATA_SET_PATH + '/train/' + this_img_filename
        this_img_array = utils.img_to_numpy(Image.open(this_img_path))

        ret_x.append(this_img_array.transpose())

        this_vec_len, this_label = this_img_filename.split('_')[0], this_img_filename.split('_')[1]

        label_list.append(this_label)

        begin_idx += 1

    ret_y = vtm.vec_to_sparse_mat(vtm.list_to_vec(label_list))

    return np.array(ret_x), ret_y


def get_test_batch(batch_size=200):
    test_path = configs.DATA_SET_PATH + '/test/'
    test_pics = os.listdir(test_path)
    np.random.shuffle(test_pics)
    if batch_size > len(test_pics):
        return None

    ret_x, label_list = [], []

    for i in range(batch_size):
        this_img_filename = test_pics[i]
        this_img_path = configs.DATA_SET_PATH + '/test/' + this_img_filename
        this_img_array = utils.img_to_numpy(Image.open(this_img_path))

        ret_x.append(this_img_array.transpose())

        this_vec_len, this_label = this_img_filename.split('_')[0], this_img_filename.split('_')[1]

        label_list.append(this_label)

    ret_y = vtm.vec_to_sparse_mat(vtm.list_to_vec(label_list))
    ret_seq_len = np.ones(batch_size, dtype=np.int32) * configs.PIC_WIDTH

    return np.array(ret_x), ret_y, ret_seq_len



if __name__ == '__main__':
    # Just for test
    a = get_train_set()
    tst_x, tst_y = get_test_batch(100)
    # print(tst_y[2])



