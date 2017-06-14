import tensorflow as tf
import timeit
g1 = tf.Graph()
with g1.device('/cpu:0'):  # 用cpu计算图
# with g1.device('/gpu:0'): # 用gpu计算图
    with g1.as_default():
        v = tf.get_variable('v', initializer=tf.zeros_initializer(), shape=[1])

# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope('', reuse=True):
#         print(sess.run(tf.get_variable('v')))

a = tf.constant([1.0, 2.0, -2.0], name='a')
b = tf.constant([1.0, 7.0, 2.0], name='b')
c = tf.constant([1.0, 7.0, 2.0], name='b')
add = tf.add(a, b, name='add')
add2 = tf.add(a, c, name='add--')

s = timeit.timeit()
with tf.Session() as sess:
    print(add)
    print(add2)
e = timeit.timeit()
print(e-s)