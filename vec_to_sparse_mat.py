# -*- coding:utf-8 -*-
import numpy as np
import configs


def list_to_vec(label_list, max_len=configs.MAX_LEN):
    size = len(label_list)
    vec = np.ones((size, max_len), dtype=np.int32)
    vec *= -1

    for i in range(size):
        label = label_list[i]
        for j in range(len(label)):
            vec[i, j] = configs.CHARS_SET.index(label[j])
    return vec


def vec_to_sparse_mat(vec):
    num_feat, num_len = vec.shape[0], vec.shape[1]
    indices = []
    values = []
    dense_shape = (num_feat, num_len)

    for i in range(num_feat):
        for j in range(num_len):
            if vec[i, j] != -1:
                indices.append([i, j])
                values.append(vec[i, j])

    return indices, values, dense_shape


def sparse_mat_to_vec(indices, values, dense_shape):
    vec = np.ones(dense_shape, dtype=np.int32) * -1
    for i in range(len(values)):
        vec[indices[i][0], indices[i][1]] = values[i]

    return vec


def decode_sparse_tensor(sparse_tensor):
    shape = tuple(sparse_tensor[2])
    vals = list(sparse_tensor[1])
    indices = list(sparse_tensor[0])
    nd_vec = sparse_mat_to_vec(indices, vals, shape)

    shape = nd_vec.shape

    decoded_list = []

    for i in range(shape[0]):
        label = ''
        for j in range(shape[1]):
            if nd_vec[i, j] == -1:
                break
            else:
                label += configs.CHARS_SET[nd_vec[i, j]]
        decoded_list.append(label)

    return decoded_list


def calc_sparse_tensor_acc(true_tensor, pred_tensor):
    true_list = decode_sparse_tensor(true_tensor)
    pred_list = decode_sparse_tensor(pred_tensor)

    if len(true_list) != len(pred_list):
        return None
    else:
        acc_cnt = 0
        for i in range(len(true_list)):
            if true_list[i] == pred_list[i]:
                acc_cnt += 1
        return acc_cnt / len(true_list)


if __name__ == '__main__':
    a = list_to_vec(['123z', '2x', 'd12'])
    b = vec_to_sparse_mat(a)
    c = decode_sparse_tensor(b)
    d = calc_sparse_tensor_acc(b, b)
    # ['123z', '2x', 'd12'] 1.0
    print(c, d)
