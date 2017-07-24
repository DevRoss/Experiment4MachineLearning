# coding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os

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
