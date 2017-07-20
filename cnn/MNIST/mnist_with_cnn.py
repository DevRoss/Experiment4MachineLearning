# coding: utf-8
import tensorflow as tf

# 神经网络
INPUT_NODE = 784
OUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# First conv size and depth
CONV1_SIZE = 5
CONV1_DEPTH = 32

# Second conv size and depth
CONV2_SIZE = 5
CONV2_DEPTH = 64

# Full connected neural network
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    # First cnn layer for convolution
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.Variable(tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEPTH], stddev=0.1),
                                   name='conv1_weight')
        conv1_biases = tf.Variable(tf.constant(0.0, shape=[CONV1_DEPTH]), name='conv1_biases')
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # Second cnn layer for pooling
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Third cnn layer for convolution
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.Variable(tf.truncated_normal([CONV2_SIZE, CONV1_SIZE, CONV1_DEPTH, CONV2_DEPTH], stddev=0.1),
                                   name='conv2_weight')
        conv2_biases = tf.Variable(tf.constant(0.0, shape=[CONV1_DEPTH]), name='conv2_biases')
        conv2 = tf.nn.conv2d(pool1, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
