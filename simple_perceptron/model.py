# coding: utf-8
import tensorflow as tf
import numpy as np

# np.random.seed(300)
learning_rate = 0.1
n_input = 2
n_output = 1

data_x = np.random.random((50, n_input))  # [50, 5] input
data_y = [[int(x1 + x2 < 1)] for x1, x2 in data_x]

test_x = np.random.random((50, n_input))
test_y = [[int(x1 + x2 < 1)] for x1, x2 in test_x]

print(np.sum(data_y))
x = tf.placeholder(tf.float32, shape=[None, n_input], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, n_output], name='y')

# 两层神经网络
w1 = tf.Variable(tf.truncated_normal(shape=[n_input, 3], stddev=1.0))
b1 = tf.Variable(tf.constant(1.0, shape=[3]))
w2 = tf.Variable(tf.truncated_normal(shape=[3, n_output]))
b2 = tf.Variable(tf.constant(1.0, shape=[n_output]))

# 向前传播
z1 = tf.matmul(x, w1)
a1 = tf.nn.bias_add(z1, b1)
a2 = tf.matmul(a1, w2)
h = tf.nn.bias_add(a2, b2)

# 二分类使用sigmoid来正则化，并计算损失
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=h))
# 使用梯度下降来优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    STEP = 500
    for i in range(STEP):
        sess.run(train_step, feed_dict={x: data_x, y_: data_y})

        if i % 1000 == 0:
            total_loss = sess.run(loss, feed_dict={x: data_x, y_: data_y})
            print('loss', total_loss)

    prediction = sess.run(h, feed_dict={x: test_x})
    prediction = prediction > 0  # >0 表示正类，<0 表示负类
    print(np.mean(np.equal(test_y, prediction)))
