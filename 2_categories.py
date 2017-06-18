import tensorflow as tf
from numpy.random import RandomState

# batch 大小
batch_size = 5

# 神经网络参数， 向前传播
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=1))

# placeholder
# shape 第一项设为None 可以在一个维度上使用不同大小的batch， 一次性使用全部数据，
# 如果数据较大，一次性法如数据可能会造成内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 神经网络向前传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数（目标值和真实值的差距）
# https://www.tensorflow.org/versions/master/api_docs/python/tf/clip_by_value
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))  # 交叉熵

# train_step 定义优化方法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成一个随机数集

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# print(X)
# 给样本定义标签， x1+x2 < 1的样本为正样本
Y = [[int(x1 + x2 < 1)] for x1, x2 in X]

# 初始化函数
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # 训练次数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选batch_size 个数据
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,
                                           feed_dict={x: X, y_: Y})
            print('step:%d \nentropy:%g'
                  %(i, total_cross_entropy))
