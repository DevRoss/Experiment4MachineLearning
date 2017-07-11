import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据
INPUT_NODE = 784  # 图片784个像素点
OUTPUT_NODE = 10  # 0~9 数字的结果

# 神经网络
LAYER1_NODE = 500  # 一层有500个节点
BATCH_SIZE = 100  # batch size
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化损失函数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 如果没有提供滑动平均类， 直接使用参数当前的取值
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 隐藏层
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 输出层
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不使用滑动平均模型
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 训练的时候不会被修改变量
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    # 将滑动平均应用到全部变量中
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 当只有分类一个正确结果的时候，可以选择使用sparse_softmax_cross_entropy_with_logits
    # 需要用argmax来得到正确答案的编号
    # tf.argmax(y_, 1) 的 1 表示在第一个维度中选取
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边的权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率，学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练所需要的轮数
        LEARNING_RATE_DECAY  # 学习率的衰减率
    )
    # 梯度下降那个优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # 将反向传播和计算滑动平均绑定在一起，这样就可以一次完成多个操作
    # 一下代码和  train_op = tf.group([train_step, variables_averages_op]) 等价
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 判断预测是否准确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 将布尔转换为实数，在求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 验证数据
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 测试数据
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # mnist 数据较小，可以一次放入
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))

            # 产生这一轮的一个 batch 的训练数据， 并进行训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束后，验证正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), validation accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    # mnist 数据包目录位置
    data_dir = os.path.join(os.path.abspath(os.path.curdir), 'MNIST_data')
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    train(mnist)


# Tensorflow 提供一个主程序入口， tf.app.run() 会调用main

if __name__ == '__main__':
    tf.app.run()
