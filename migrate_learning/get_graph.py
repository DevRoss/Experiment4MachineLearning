import tensorflow as tf
from tensorflow.python.platform import gfile
import os
from data_collection import get_image, get_flower
import numpy as np
import random

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


# 随机读取一个batch 作为训练数据
def get_random_cached_bottlenecks(sess, n_class, result, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = list()
    ground_truths = list()
    for _ in range(how_many):
        label_index = random.randrange(n_class)
        label_name = list(result.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, result, label_name, image_index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(n_class, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def main(_):
    result = get_flower(TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)('flower_photos')
    n_class = len(result.keys())
    # 读取模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,
                                                              return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                               JPEG_DATA_TENSOR_NAME])

    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_class], name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_class], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_class]))
        logits = tf.matmul(bottleneck_tensor, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 交叉熵作为损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_class, result, BATCH,
                                                                                  'training', jpeg_data_tensor,
                                                                                  bottleneck_tensor)

            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,
                                            ground_truth_input: train_ground_truth})

            # 验证正确率
            if i % 100 == 0:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_class, result,
                                                                                                BATCH,
                                                                                                'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                                           ground_truth_input: validation_ground_truth})

                print('Step %d Validation accuracy on random sampled %d examples = %.lf%%' % (
                    i, BATCH, validation_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()