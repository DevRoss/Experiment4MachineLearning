import tensorflow as tf

# filter layer
filter_weight = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 16], stddev=0.1), name='filter_weight')
# biases
biases = tf.Variable(tf.constant(0.1), name='biases')

# convolution
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# biases
bias = tf.nn.bias_add(conv, biases)

# excitement function
actived_conv = tf.nn.relu(bias)