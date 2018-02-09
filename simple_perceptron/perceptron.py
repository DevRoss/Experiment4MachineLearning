# coding: utf-8
from sklearn.neural_network import MLPClassifier
import numpy as np
from keras.utils import np_utils
import os
import logging
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

logger = logging.getLogger()
data_dir = '../smp2017_5fold'  # 数据文件夹

np_x_train_data = '{fold}_train_test_x_train_kf_smp_bow.npy'
np_y_train_data = '{fold}_train_test_y_train_kf_smp_bow.npy'
np_x_test_data = '{fold}_train_test_x_test_kf_smp_bow.npy'
np_y_test_data = '{fold}_train_test_y_test_kf_smp_bow.npy'

# 最大词数
MAX_LEN = 20
# 每个词映射的维度
WORD_DIMENSION = 50


def Cross_validation(x_train, y_train, x_test, y_test, hidden_layer_sizes, alpha, solver,
                     max_iter) -> float:
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha, max_iter=max_iter)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


if __name__ == '__main__':
    logger.setLevel(logging.INFO)

    hidden_layer_sizes = [[100, 100], [100, 200], [100, 250], [200, 100], [200, 200], [200, 250], [250, 100],
                          [250, 200], [250, 250], [300, 300]]

    max_iter = 400
    solvers = ['adam', 'sgd']
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.6]
    output_file = 'MLP.csv'
    logger.info('Running...')
    result = []
    for solver in solvers:
        logger.info('Running' + solver)
        for hidden_layer_size in hidden_layer_sizes:
            for alpha in alphas:
                for fold in range(0, 5):
                    # x_train = pad_sequences(np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold))), 20)
                    # x_train = x_train.reshape((len(x_train), -1))
                    x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                    y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))
                    # x_test = pad_sequences(np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold))), 20)
                    # x_test = x_test.reshape((len(x_test), -1))
                    x_test = np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold)))
                    y_test = np.load(os.path.join(data_dir, np_y_test_data.format(fold=fold)))

                    y_train = np_utils.to_categorical(y_train, 31)  # 31个分类
                    y_test = np_utils.to_categorical(y_test, 31)
                    print(x_train.shape)
                    print(y_train.shape)
                    acc = Cross_validation(x_train, y_train, x_test, y_test, hidden_layer_size, alpha, solver, max_iter)
                    result.append((acc, hidden_layer_size, alpha, max_iter, solver))
    logger.info('Model training done.')
    logger.info('Writing results to' + output_file)
    df = pd.DataFrame(data=result, columns=['accuracy', 'hidden_layer_size', 'alpha', 'max_iter', 'solver'])
    df.to_csv(output_file, index=False)
    logger.info('Done.')
