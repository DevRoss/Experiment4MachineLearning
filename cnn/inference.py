import tensorflow as tf

# filter layer
filter_weight = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 16], stddev=0.1), name='filter_weight')
# biases
biases = tf.Variable(tf.constant(0.1), name='biases')
