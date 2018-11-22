#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pic_model
import configs
import numpy as np
import utils
from PIL import Image, ImageFilter
import vec_to_sparse_mat as vtm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def import_model(model_path, save_path, sess):
    importer = tf.train.import_meta_graph(model_path)
    importer.restore(sess, tf.train.latest_checkpoint(save_path))


def nd_array_pic_show(array, title):
    Image.fromarray(array).show(title=title)


def gen_and_show_test_pic(ft, if_show=False):
    length = np.random.randint(1, configs.MAX_LEN + 1)
    label = ""
    for j in range(length):
        idx = np.random.randint(0, len(configs.CHARS_SET))
        label += configs.CHARS_SET[idx]
    image = ft.draw_text(utils.gen_background_image(), configs.TEXT_POS, label, \
                         configs.FONT_SIZE, utils.gen_random_color())
    image = Image.fromarray(image)
    grey_img = image.convert('L')  # 先转成灰度值
    bin_img = grey_img.point(configs.BIN_TABLE, '1')  # 得到二值化后的黑白图片
    fil_img = bin_img.filter(ImageFilter.MedianFilter(size=9))

    ret_sparse = vtm.vec_to_sparse_mat(vtm.list_to_vec([label]))

    # 展示图片处理过程
    if if_show:
        image.show(title='Original')
        grey_img.show(title='Grey')
        bin_img.show(title='Binary')
        fil_img.show(title='Filtered')

    return np.array(fil_img, dtype=np.int32).transpose().\
        reshape([1, configs.PIC_WIDTH, configs.PIC_HEIGHT]), ret_sparse


def gen_one_test(if_show=False):
    ft = pic_model.pic_model()
    pic_array, pic_label = gen_and_show_test_pic(ft, if_show=if_show)
    return pic_array, pic_label


if __name__ == '__main__':
    with tf.Session() as sess:
        import_model('./models/ocr.model-50000.meta', './models', sess)
        graph = tf.get_default_graph()
        inputs = graph.get_operation_by_name('inputs').outputs[0]

        class Targets:
            def __init__(self):
                self.shape = None
                self.values = None
                self.indices = None

        targets = Targets()
        targets.shape = graph.get_operation_by_name('targets/shape').outputs[0]
        targets.values = graph.get_operation_by_name('targets/values').outputs[0]
        targets.indices = graph.get_operation_by_name('targets/indices').outputs[0]

        seq_len = graph.get_operation_by_name('seq_len').outputs[0]

        logits = tf.get_collection('logits')[0]
        acc = tf.get_collection('acc')[0]

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        test_seq_len = np.ones(1, dtype=np.int32) * configs.PIC_WIDTH

        while True:
            k = input('Put Q to exit, any other key to continue...')
            if k == 'Q':
                break
            else:
                test_input, test_targets = gen_one_test(if_show=True)
                feed_dict = {
                    inputs: test_input,
                    targets.shape: test_targets[2],
                    targets.values: test_targets[1],
                    targets.indices: test_targets[0],
                    seq_len: test_seq_len
                }
                pred_sparse_tensor, log_probs, accuracy = sess.run([decoded[0], log_prob, acc], feed_dict)
                true_label, pred_label = vtm.decode_sparse_tensor(test_targets)[0], \
                    vtm.decode_sparse_tensor(pred_sparse_tensor)[0]
                print("Truth Value: %s, Predict Value: %s. " % (true_label, pred_label), end='')
                if true_label == pred_label:
                    print("Correct!")
                else:
                    print("Wrong!")
