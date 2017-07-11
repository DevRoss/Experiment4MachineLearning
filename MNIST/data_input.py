from tensorflow.examples.tutorials.mnist import input_data
import os
# MNIST 数据文件夹
data_dir = os.path.join(os.path.abspath(os.path.curdir), 'MNIST_data')
mnist = input_data.read_data_sets(data_dir, one_hot=True)

print('Training data size:', mnist.train.num_examples)
print('Validating data size:', mnist.validation.num_examples)
print('Test data size:', mnist.test.num_examples)
print('Example training data: :', mnist.train.images[0])
print('Example training data label: ', mnist.train.labels[0])

