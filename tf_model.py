# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import configs
import time
import batch
import vec_to_sparse_mat as vtm
import os
# remove tf_compile improvement warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_train_model():
    # 输入格式 BATCH_SIZE * PIC_WIDTH * PIC_HEIGHT
    # 对应 [batch_size,max_time_step,num_features] 的格式要求
    inputs = tf.placeholder(tf.float32, [None, configs.PIC_SHAPE[1], configs.PIC_SHAPE[0]], name='inputs')

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32, name='targets')

    # 1维向量 序列长度 [batch_size] 值为max_time_step 即 PIC_WIDTH
    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(configs.NUM_HIDDEN, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)

    # 获取输入的形状
    shape = tf.shape(inputs)

    # [batch_size, PIC_WIDTH]
    batch_size, max_time_steps = shape[0], shape[2]

    # [batch_size*max_time_step, num_hidden] 以便进行后面的matmul计算
    outputs = tf.reshape(outputs, [-1, configs.NUM_HIDDEN])

    # 定义权重W与偏差b
    W = tf.Variable(tf.truncated_normal([configs.NUM_HIDDEN,
                                         configs.NUM_CLASSES],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[configs.NUM_CLASSES]), name="b")

    # [batch_size*max_time_steps, num_classes]
    logits = tf.matmul(outputs, W) + b

    # [batch_size, max_time_steps, num_classes]
    logits = tf.reshape(logits, [batch_size, -1, configs.NUM_CLASSES])

    # 转置矩阵，第0和第1列互换位置 => [max_time_steps, batch_size, num_classes]
    logits = tf.transpose(logits, (1, 0, 2))

    # 定义损失函数
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    cost = tf.reduce_mean(loss)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    # Accuracy: label error rate
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    tf.add_to_collection('logits', logits)
    tf.add_to_collection('acc', acc)
    tf.add_to_collection('log_prob', log_prob)

    return logits, inputs, targets, seq_len, W, b, loss, cost, decoded, log_prob, acc


def train():
    global_step = tf.Variable(0, trainable=False)

    # 采用按指数衰减动态变化的学习率
    learning_rate = tf.train.exponential_decay(configs.LEARNING_RATE,
                                               global_step,
                                               configs.DECAY_STEPS,
                                               configs.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)

    logits, inputs, targets, seq_len, W, b, loss, cost, \
        decoded, log_prob, acc = get_train_model()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).\
        minimize(loss, global_step=global_step)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # Initialize the weights and biases
    init = tf.global_variables_initializer()

    new_config_prototype = tf.ConfigProto()
    new_config_prototype.gpu_options.allow_growth = True

    with tf.Session(config=new_config_prototype) as session:
        session.run(init)
        train_set = batch.get_train_set()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_seq_len = np.ones(configs.BATCH_SIZE, dtype=np.int32) * configs.PIC_WIDTH

        try:
            for curr_epoch in range(configs.NUM_EPOCHS):
                s_time = time.time()
                s_iter = 0
                print("Epoch %d start training....." % curr_epoch)
                while True:
                    train_x, train_y = batch.get_next_batch(train_set, s_iter, batch_size=configs.BATCH_SIZE)
                    s_iter += 1
                    if train_x is None:
                        # 没有下一个训练batch了
                        break
                    feed_dict = {
                        inputs: train_x,
                        targets: train_y,
                        seq_len: train_seq_len
                    }

                    b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
                        [loss, targets, logits, seq_len, cost, global_step, optimizer],
                        feed_dict)
                    if s_iter % configs.REPORT_STEPS == 0:
                        print("Total_step:%d Current_step: %d Train_cost: %.3f" % (steps, s_iter, b_cost))

                # 该epoch训练完毕，使用test集做一次测试
                test_x, test_y, test_seq_len = batch.get_test_batch(batch_size=configs.NUM_PICS_TEST)
                s_time = int(time.time())-int(s_time)
                if test_x is not None:
                    feed_dict = {
                        inputs: test_x,
                        targets: test_y,
                        seq_len: test_seq_len
                    }
                    pred_sparse_tensor, log_probs, accuracy = session.run([decoded[0], log_prob, acc], feed_dict)
                    curr_acc = vtm.calc_sparse_tensor_acc(test_y, pred_sparse_tensor)
                    if curr_acc is None:
                        print("Epoch: %d Length not equal." % curr_epoch, end='')
                    else:
                        print("Epoch: %d Validation Accuracy: %.2f%%." % (curr_epoch, curr_acc * 100), end='')
                        if curr_acc >= 0.995:
                            # 准确率高于99.5%，保存模型并停止训练
                            model_path = saver.save(session, "models/ocr.model", global_step=steps)
                            print('\nTraining finished. The model has saved in %s* files.' % model_path)
                            return
                    print(" Spend Time: %d s." % s_time)
        except KeyboardInterrupt:
            k = input("Detected Keyboard Interrupt. Do you want to save model? [y/n]")
            if k == 'y':
                model_path = saver.save(session, "models/ocr.model", global_step=steps)
                print('\nThe model has saved in %s* files.' % model_path)
            return


if __name__ == '__main__':
    train()
