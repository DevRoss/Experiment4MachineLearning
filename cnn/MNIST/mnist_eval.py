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
                           [None,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS], name='x-input')

        y_ = tf.placeholder(tf.float32, [None, OUT_NODE], name='y-input')
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

        # 直接使用滑动平均值来恢复变量
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                # 加载模型
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 训练轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 正确率
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print('After %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint found.')
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()