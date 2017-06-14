import tensorflow as tf

# weight

w1 = tf.Variable(initial_value=tf.random_normal([2, 3], stddev=1, seed=1))  # 生成2x3的矩阵， 标准差为1
w2 = tf.Variable(initial_value=tf.random_normal([3, 1], stddev=1, seed=1))  # 生成3x1的矩阵， 标准差为1

# input

x = tf.constant([[0.7, 0.9]])   # 输入为一个1x2的矩阵

# spread
a = tf.matmul(x, w1)    # 第一层
y = tf.matmul(a, w2)    # 第二层

# Tensorflow Session

with tf.Session() as sess:
    sess.run(w1.initializer)    # 初始化w1
    sess.run(w2.initializer)    # 初始化w2
    print(sess.run(y))