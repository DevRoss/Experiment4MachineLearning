from .mnist_with_cnn import *
import tensorflow as tf
import os

BATCH_SIZE = 100
REGULARIZATION_RATE = 0.0001  # 正则化损失函数的
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
TRAINING_STEPS = 30000  # 训练轮数

# 模型保存
MODEL_SAVE_PATH = os.path.join(os.path.abspath(os.path.curdir), 'model')
MODEL_NAME = 'mnist'


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

    # Train with gradient descent
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean, global_step)


    saver = tf.train.Saver()
    init_op =tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

    for i in range(TRAINING_STEPS):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)

        train_op, loss_value, step = sess.run([train_step, cross_entropy_mean, global_step],
                                              feed_dict={x:xs, y_:ys})
        if i % 1000 == 0:
            # 每1000次保存一次模型
            print('After %d training step(s), loss on training batch is %g' % (step, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)