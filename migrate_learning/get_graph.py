import tensorflow as tf
import os

# Inception-v3 模型瓶颈层节点数
BOTTLENECK_TENSOR_SIZE = 2048

# 瓶颈层 tensor 名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 输入 tensor
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# Inception v3 目录
MODEL_DIR = 'inception_dec_2015'

# 模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 训练数据的特征向量
CACHE_DIR = 'tmp/bottleneck'

# flower 文件夹名字
INPUT_DATA = 'flower_photos'

# 各种数据百分比
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# 神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100