import tensorflow as tf

'''
查看model.ckpt 的 variable 和 shape
'''
reader = tf.train.NewCheckpointReader('model/model.ckpt')
all_variables = reader.get_variable_to_shape_map()
for v_name in all_variables:
    print(v_name, all_variables[v_name])
    print('Value is %s' % reader.get_tensor(v_name))
