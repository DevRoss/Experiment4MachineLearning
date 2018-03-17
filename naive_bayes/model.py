# coding: utf-8
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from keras.utils import np_utils
import os
import logging
import pandas as pd

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

data_dir = '../smp2017_5fold'  # 数据文件夹

np_x_train_data = '{fold}_train_test_x_train_kf_smp_bow.npy'
np_y_train_data = '{fold}_train_test_y_train_kf_smp_bow.npy'
np_x_test_data = '{fold}_train_test_x_test_kf_smp_bow.npy'
np_y_test_data = '{fold}_train_test_y_test_kf_smp_bow.npy'

# 最大词数
MAX_LEN = 20
# 每个词映射的维度
WORD_DIMENSION = 50


def Cross_validation(x_train, y_train, x_test, y_test, alpha, fit_prior) -> float:
    model = MultinomialNB(alpha, fit_prior)  # 31个类
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


if __name__ == '__main__':
    alpha_set = [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
    fit_prior_options = [True, False]
    output_file = 'NaiveBayes.csv'
    logger.info('Running...')
    result = []
    for alpha in alpha_set:
        logger.info('Running' + alpha.__str__())
        for fit_prior in fit_prior_options:
            for fold in range(0, 5):
                x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))

                x_test = np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold)))
                y_test = np.load(os.path.join(data_dir, np_y_test_data.format(fold=fold)))

                # y_train = np_utils.to_categorical(y_train, 31)  # 31个分类
                # y_test = np_utils.to_categorical(y_test, 31)

                acc = Cross_validation(x_train, y_train, x_test, y_test, alpha, fit_prior)
                result.append((acc, fold, alpha, fit_prior))
    logger.info('Model training done.')
    logger.info('Writing results to' + output_file)
    df = pd.DataFrame(data=result, columns=['accuracy', 'fold', 'alpha', 'fit_prior'])
    df.to_csv(output_file, index=False)
    logger.info('Done.')
