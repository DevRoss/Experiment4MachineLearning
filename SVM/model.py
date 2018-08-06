# -*- coding: utf-8 -*-
# @Time    : 2018/5/17 21:52
# @Author  : Ross
# @File    : model.py
from __future__ import print_function
import logging
import numpy as np
from keras.utils import np_utils
from sklearn.svm import SVC
import pickle
import pandas as pd
import os

data_dir = '../SMP2017/smp2017_5fold'
output_file = 'SVM.csv'

np_x_train_data = "{fold}train_x_fold.npy"
np_y_train_data = "{fold}train_y_fold.npy"
np_x_dev_data = "{fold}val_x_fold.npy"
np_y_dev_data = "{fold}val_y_fold.npy"
np_x_test_data = "smp_test_x_bow.pkl"
np_y_test_data = "smp_test_y.pkl"
with open(os.path.join(data_dir, np_x_test_data), "rb") as f:
    x_test = pickle.load(f)
with open(os.path.join(data_dir, np_x_test_data), "rb") as f:
    y_test = pickle.load(f)

# 最大词数
maxlen = 20
# 每个词映射的维度
len_wv = 50


def Cross_validation(x_train, y_train, x_dev, y_dev, x_test, C, \
                     kernel, gamma, decision_function_shape):
    logging.log(logging.INFO,
                'training, params: C={}, kernel={}, gamma={}, decision_function_shape={}'.format(C, kernel, gamma,
                                                                                                 decision_function_shape))
    model = SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape=decision_function_shape, probability=True)
    model.fit(x_train, y_train)
    # 对测试集进行预测
    tmp = model.predict_proba(x_test)
    accuracy = model.score(x_dev, y_dev)
    # 输出当折最佳acc和对test的预测
    return tmp, accuracy


def countAcc(y_pre, y_true):
    sumT = 0
    for i in range(len(y_pre)):
        predict_label = np.array(y_pre[i]).argsort()[-1]
        # true_label = np.array(y_true[i]).argsort()[-1]
        # print(predict_label,true_label)
        if predict_label == y_true[i]:
            sumT += 1
    return sumT / (len(y_true) + 0.0000000000000001)

    # k = np.array(x_pre[i]).argsort()[-1:-2:-1]


if __name__ == '__main__':
    result = []
    C_ = [1.65]
    kernels = ['linear']
    gammas = ['auto']
    decision_function_shapes = ['ovo']
    for C in C_:
        for kernel in kernels:
            for gamma in gammas:
                for decision_function_shape in decision_function_shapes:
                    predict = list()
                    acc = 0.0
                    # y_test = np_utils.to_categorical(y_test, 31)
                    try:
                        for fold in range(0, 5):
                            x_train = np.load(os.path.join(data_dir, np_x_train_data.format(fold=fold)))
                            y_train = np.load(os.path.join(data_dir, np_y_train_data.format(fold=fold)))

                            x_dev = np.load(os.path.join(data_dir, np_x_dev_data.format(fold=fold)))
                            y_dev = np.load(os.path.join(data_dir, np_y_dev_data.format(fold=fold)))
                            # y_dev = np_utils.to_categorical(y_dev, 31)  # 必须使用固定格式表示标签
                            # y_train = np_utils.to_categorical(y_train, 31)  # 必须使用固定格式表示标签 一共 31分类

                            predict_tmp, acc_tmp = Cross_validation(x_train, y_train, x_dev, y_dev, x_test,
                                                                    C, kernel, gamma, decision_function_shape)
                            if len(predict) == 0:
                                predict = predict_tmp
                            else:
                                predict += predict_tmp
                            acc += acc_tmp
                            print(acc_tmp)
                        acc = acc / 5
                        # 计算集成测试的正确率
                        test_acc = countAcc(predict, y_test)

                        result.append((acc, test_acc, C, kernel, gamma, decision_function_shape))
                        print('Test accuracy:', acc)

                    except Exception as e:
                        print(e)
logging.log(logging.INFO, 'Write to csv')
df = pd.DataFrame(data=result,
                  columns=['accuracy', 'test_acc', 'C', 'kernel', 'gamma', 'decision_function_shape'])

df.to_csv(output_file, index=False)
logging.log(logging.INFO, 'done')
