import tensorflow as tf

v1 = tf.Variable(tf.random_normal(shape=[2], stddev=0.1), dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(0.98)

# avg = ema.apply(tf.all_variables())
# saver = tf.train.Saver({'v1': v1})

# 保存的时候变成保存滑动平均值
saver = tf.train.Saver(ema.variables_to_restore())
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # saver.save(sess, 'saver/model.ckpt')
    # saver.restore(sess, 'saver/model.ckpt')
    # 以json格式导出
    saver.export_meta_graph('saver/model.json', as_text=True)
    print(sess.run(v1))
    for _ in tf.all_variables():
        print(_.name)
