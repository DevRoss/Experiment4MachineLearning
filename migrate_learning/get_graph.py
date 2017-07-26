import tensorflow as tf
from tensorflow.python.platform import gfile
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
    image_path = get_image(result, label_name, index, category)
    basename = os.path.basename(image_path)
    cache_file = os.path.join(CACHE_DIR, label_name, basename + '.txt')
    return cache_file


# 传入图片，后的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # 经过卷积后变成一个四维数组，将其压缩成一个一维向量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 缓存机制，如果特征向量已经保存，则返回特征向量，否则先存入文件，再返回特征向量。
def get_or_create_bottleneck(sess, result, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = result[label_name]
    sub_dir_path = os.path.join(CACHE_DIR, label_name)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(result, label_name, index, category)

    # 如果特征向量文件不存在
    if not os.path.exists(bottleneck_path):
        image_path = get_image(result, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 获取特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else:
        # 从文件中读取特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values