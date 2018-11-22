# -*- coding:utf-8 -*-

# 文件相关参数
NUM_PICS_TEST = 400  # 测试集图片数目
NUM_PICS_TRAIN = 1600  # 训练集图片数目
DATA_SET_PATH = './data'

# 图片参数
PIC_HEIGHT = 60
PIC_WIDTH = 160
PIC_SHAPE = (PIC_HEIGHT, PIC_WIDTH)
TEXT_POS = (10, 10)
FONT_SIZE = 54

MAX_LEN = 4  # 变长验证码的最大长度
CHARS_NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CHARS_UPPER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
CHARS_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHARS_SET = CHARS_NUMBER + CHARS_UPPER + CHARS_LOWER

FONT_PATH = './fonts/OCR-B.ttf'

# 训练最大轮次
NUM_EPOCHS = 10000

# 图片大小为60*160
BATCH_SIZE = 64
# 字符备选集大小 + CTC Blank
NUM_CLASSES = len(CHARS_SET) + 1

# LSTM
NUM_HIDDEN = 64
NUM_LAYERS = 1

# 超参数
LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 5  # 每n步 做一次测试集报告
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

# 图片处理相关
BIN_THRESHOLD = 220
BIN_TABLE = []

for i in range(256):
    if i < BIN_THRESHOLD:
        BIN_TABLE.append(0)
    else:
        BIN_TABLE.append(1)

