from .mnist_with_cnn import *
import tensorflow as tf

BATCH_SIZE = 100

def train(mnist):
    # 图片输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           [BATCH_SIZE,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_NODE], name='y-input')

