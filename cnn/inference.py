import tensorflow as tf

# 输入层
'''
如果是输入层 [a, b, c, d]中a表示batch， b c d 分别表示的的是 输入图片的长、宽和深度
'''
input_layer = tf.placeholder(tf.float32, [None, 32, 32, 3], name='image_input')

# filter layer
'''
过滤器，shape 表示  5x5 过滤器，(3表示当前深度，即表示 RGB三原色)，16 为输出深度
'''
filter_weight = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 16], stddev=0.1), name='filter_weight')
# biases
'''
偏置项的 shape 应是输出深度
'''
biases = tf.Variable(tf.constant(0.1, shape=[16]), name='biases')

# convolution
'''
卷积层向前传播
strides: 表示不同维度的步长，第一个和第四个必须为1才能使得传播有效，
padding: tensorflow 提供了 'SAME' 和 'VALID' 两个选择，SAME 表示填充0，VALID表示不填充
'''
conv = tf.nn.conv2d(input_layer, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# biases
'''
不能直接添加biases，因为需要为每一个节点都调价biases
例如： 将3x3 的图片卷积成 2x2，需要为2x2的矩阵每一个值都添加biases
'''
bias = tf.nn.bias_add(conv, biases)

# excitement function
'''
激活函数
'''
actived_conv = tf.nn.relu(bias)

# pooling
'''
pooling 和 convolution 的作用对象不同， convolution 作用对象包括深度在内的三维矩阵（立方体），
pooling作用对象是深度为二维矩形，即深度为1矩形

pool 有助于防止过拟合问题，和加快计算速度
ksize: kernel size 过滤器尺寸，和strides一样第一项和第四项要为1的时候才起作用
strides: 和卷积层一样，表示不同维度的步长，第一个和第四个必须为1才能使得传播有效
padding: 和卷积层一样，tensorflow 提供了 'SAME' 和 'VALID' 两个选择，SAME 表示填充0，VALID表示不填充
'''
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
