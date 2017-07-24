# coding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.curdir))

# 加载常量
from mnist_train import *
from mnist_with_cnn import *

# 验证的时间频率, 10 秒进行一次验证
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [BATCH_SIZE,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS], name='x-input')

        y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_NODE], name='y-input')
        # 改变维度
        xs = np.reshape(mnist.validation.images,
                        [mnist.validation.num_examples,
                         IMAGE_SIZE,
                         IMAGE_SIZE,
                         NUM_CHANNELS
                         ])
        ys = mnist.validation.labels
        validate_feed = {x: xs, y_: ys}

        # 向前传播得到结果
        y = inference(xs)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))