# coding: utf-8
from sklearn.linear_model import LogisticRegression
import logging
import pandas as pd
import numpy as np
import os

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


def Cross_validation(x_train, y_train, x_test, y_test, penalty, c, fit_intercept, solver, multi_class) -> float:
    model = LogisticRegression(penalty=penalty, C=c, fit_intercept=fit_intercept, solver=solver,
                               multi_class=multi_class)  # 31个类
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


if __name__ == '__main__':
    penalties = ['l1', 'l2']
    cs = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    fit_intercept_options = [True, False]
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    max_iters = [100, 150, 200]
    multi_classes = ['ovr', 'multinomial']

    output_file = 'LogisticRegression.csv'
    logger.info('Running...')
    result = []
    for penalty in penalties:
        for c in cs:
            for fit_intercept in fit_intercept_options:
                for solver in solvers:
                    for max_iter in max_iters:
                        for multi_class in multi_classes:
                            if solver in ['newton-cg', 'sag', 'saga'] and penalty == 'l1':
                                continue
                            if solver == 'lbfgs' and penalty == 'l1':
                                continue
                            if solver == 'liblinear' and multi_class == 'multinomial':
                                continue
                            for fold in range(0, 5):
                                x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                                y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))

                                x_test = np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold)))
                                y_test = np.load(os.path.join(data_dir, np_y_test_data.format(fold=fold)))

                                # y_train = np_utils.to_categorical(y_train, 31)  # 31个分类
                                # y_test = np_utils.to_categorical(y_test, 31)

                                acc = Cross_validation(x_train, y_train, x_test, y_test, penalty, c, fit_intercept,
                                                       solver, multi_class)
                                result.append((acc, fold, penalty, c, fit_intercept,
                                               solver, multi_class))
logger.info('Model training done.')
logger.info('Writing results to' + output_file)
df = pd.DataFrame(data=result,
                  columns=['accuracy', 'fold', 'penalty', 'c', 'fit_intercept',
                           'solver', 'multi_class'])
df.to_csv(output_file, index=False)
logger.info('Done.')
