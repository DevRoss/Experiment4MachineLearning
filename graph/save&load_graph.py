import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

'''
保存训练好的模型，不用保存类似于变量初始化，模型保存等的辅助节点
'''
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    # 到处当前计算图的GraphDef 部分，只需要这部分就可以完成向前传播的得到输出层
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量取常量，同时去掉类似于初始化的不必要的节点。
    # 后面的 ['add'] 参数给出了保存的节点的名称
    # 后面没有 :0 是因为add节点定义的只是两个变量相加的操作，而不是张量

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add']
    )

    # # 写入文件，记得先新建文件夹
    # with tf.gfile.GFile('model/model.pb', 'wb') as f:
    #     f.write(output_graph_def.SerializeToString())

    # 读取文件
    # 通常用于迁移学习
    model_file = './graph.pb'

    if not os.path.exists(os.path.dirname(model_file)):
        os.mkdir(os.path.dirname(model_file))

    with tf.gfile.GFile(model_file, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    with tf.gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        result = tf.import_graph_def(graph_def, return_elements=['add:0'])
        print(sess.run(result))
