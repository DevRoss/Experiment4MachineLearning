# coding: utf-8
from sklearn.tree import DecisionTreeClassifier
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


def Cross_validation(x_train, y_train, x_test, y_test, criterion, splitter, min_samples_split, max_features,
                     min_impurity_decrease) -> float:
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split,
                                   max_features=max_features, min_impurity_decrease=min_impurity_decrease)  # 31个类
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc


if __name__ == '__main__':
    criterions = ['gini', 'entropy']
    splitters = ['best', 'random']
    min_samples_splits = [0.01, 0.05, 0.1, 0.2, 0.5]
    _max_features = ['auto', 'sqrt', 'log2', None]
    min_impurity_decreases = [0., 0.1, 0.2, 0.5]

    output_file = 'DecisionTree.csv'
    logger.info('Running...')
    result = []
    for criterion in criterions:
        for splitter in splitters:
            for min_samples_split in min_samples_splits:
                for max_features in _max_features:
                    for min_impurity_decrease in min_impurity_decreases:
                        for fold in range(0, 5):
                            x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                            y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))

                            x_test = np.load(os.path.join(data_dir, np_x_test_data.format(fold=fold)))
                            y_test = np.load(os.path.join(data_dir, np_y_test_data.format(fold=fold)))

                            # y_train = np_utils.to_categorical(y_train, 31)  # 31个分类
                            # y_test = np_utils.to_categorical(y_test, 31)

                            acc = Cross_validation(x_train, y_train, x_test, y_test, criterion, splitter,
                                                   min_samples_split, max_features, min_impurity_decrease)
                            result.append((acc, fold, criterion, splitter, min_samples_split, max_features,
                                           min_impurity_decrease))
    logger.info('Model training done.')
    logger.info('Writing results to' + output_file)
    df = pd.DataFrame(data=result,
                      columns=['accuracy', 'fold', 'criterion', 'splitter', 'min_samples_split', 'max_features',
                               'min_impurity_decrease'])
    df.to_csv(output_file, index=False)
    logger.info('Done.')
