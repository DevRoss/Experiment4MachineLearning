from .mnist_with_cnn import *
import tensorflow as tf

BATCH_SIZE = 100
REGULARIZATION_RATE = 0.0001  # 正则化损失函数的


def train(mnist):
    # 图片输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           [BATCH_SIZE,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_NODE], name='y-input')

    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # output layer
    y = inference(x, train=True, regularizer=regularizer)
