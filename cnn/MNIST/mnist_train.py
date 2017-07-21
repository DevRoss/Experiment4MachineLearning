from .mnist_with_cnn import *
import tensorflow as tf

BATCH_SIZE = 100
REGULARIZATION_RATE = 0.0001  # 正则化损失函数的
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率


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

    # losses
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        cross_entropy_mean =tf.reduce_mean(cross_entropy)

    # learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )