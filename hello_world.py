import tensorflow as tf

hello = tf.constant('hello, TF boy!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print(sess.run(a + b))
    print(sess.run(a * b))

c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)
add = tf.add(c, d)
mul = tf.multiply(c, d)
with tf.Session() as sess:
    print(sess.run(add, feed_dict={c: 2, d: 3}))
    print(sess.run(mul, feed_dict={c: 2, d: 4}))
mat1 = tf.constant([[3., 4.]])
mat2 = tf.constant([[2.], [2.]])
product = tf.matmul(mat1, mat2)
with tf.Session() as sess:
    result = sess.run(product)
    print('result is :', result)
