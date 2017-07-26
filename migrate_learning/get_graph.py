import tensorflow as tf
import os
from data_collection import get_image
import numpy as np

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


# 经过Inception v3 处理后特征的存放位置
def get_bottleneck_path(result, label_name, index, category):
    return get_image(result, label_name, index, category) + '.txt'


# 传入图片，后的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # 经过卷积后变成一个四维数组，将其压缩成一个一维向量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
