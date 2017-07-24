from mnist_with_cnn import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

BATCH_SIZE = 100
REGULARIZATION_RATE = 0.0001  # 正则化损失函数的
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99

# 模型保存
MODEL_SAVE_PATH = os.path.join(os.path.abspath(os.path.curdir), 'model')
MODEL_NAME = 'mnist.ckpt'

# MNIST
DATA_DIR = r'F:\project\tf_quick_start\MNIST\MNIST_data'


def train(mnist):
    # 图片输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           [BATCH_SIZE,
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                            NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_NODE], name='y-input')

    global_step = tf.Variable(0, trainable=False)
    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # output layer
    y = inference(x, train=True, regularizer=regularizer)

    # losses
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # learning rate decay
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # MovingAverage
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # Train with gradient descent
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # bind MV and train_step
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,
                                     [BATCH_SIZE,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE,
                                      NUM_CHANNELS]
                                     )
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                # 每1000次保存一次模型
                print('After %d training step(s), loss on training batch is %g' % (step, loss_value))
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)


# 主函数
def main(argv=None):
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
