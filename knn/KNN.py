# coding: utf-8
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.utils import np_utils
import os
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

data_dir = '../smp2017_5fold'  # 数据文件夹

np_x_train_data = '{fold}_train_test_x_train_kf_smp_bow.npy'
np_y_train_data = '{fold}_train_test_y_train_kf_smp_bow.npy'
np_x_test_data = '{fold}_train_test_x_test_kf_smp_bow.npy'
np_y_test_data = '{fold}_train_test_y_test_kf_smp_bow.npy'

# 最大词数
MAX_LEN = 20
# 每个词映射的维度
WORD_DIMENSION = 50


def Cross_validation(x_train, y_train, x_test, y_test, p, n_neighbors, weights, leaf_size) -> float:
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, n_jobs=-1, p=p)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


if __name__ == '__main__':
    n_neighbors_set = [1, 2, 5, 10, 15, 20]
    weights_set = ['uniform', 'distance']
    leaf_size_set = [10, 15, 20, 30, 40, 50, 80]
    p_set = [1, 2, 3, 4]
    output_file = 'KNN.csv'
    logger.info('Running...')
    result = []
    for weights in weights_set:
        logger.info('Running' + weights)
        for leaf_size in leaf_size_set:
            for n_neighbors in n_neighbors_set:
                for p in p_set:
                    for fold in range(0, 3):
                        x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                        y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))

                        x_test = np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold)))
                        y_test = np.load(os.path.join(data_dir, np_y_test_data.format(fold=fold)))

                        y_train = np_utils.to_categorical(y_train, 31)  # 31个分类
                        y_test = np_utils.to_categorical(y_test, 31)

                        acc = Cross_validation(x_train, y_train, x_test, y_test, p, n_neighbors, weights, leaf_size)
                        result.append((acc, fold, p, n_neighbors, weights, leaf_size))
    logger.info('Model training done.')
    logger.info('Writing results to' + output_file)
    df = pd.DataFrame(data=result, columns=['accuracy', 'fold', 'power', 'n_neighbors', 'weights', 'leaf_size'])
    df.to_csv(output_file, index=False)
    logger.info('Done.')
