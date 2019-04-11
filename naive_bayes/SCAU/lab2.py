#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by Ross on 19-4-4
import numpy as np
from data_helper import DataModel
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


def createVocabList(dataSet):
    vocabSet = set([])

    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取两个集合的并集
    return sorted(list(vocabSet))


# ③构建词向量。这里采用的是词集模型，即只需记录每个词是否出现，而不考虑其出现的次数。需要记录词出现的次数的叫词袋模型。

def setOfWords2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList))  # 生成零向量的array
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 单词出现则记为1
        else:
            # print('the word:%s is not in my Vocabulary!' % word)
            pass
    return returnVec  # 返回全为0和1的向量


# ④根据训练集计算概率。根据上面的公式：
#
# 正如前面分析的那样，我们只需考虑分子即可。
# 用训练集中，属于类别的样本数量除以总的样本数量即可；
# 可以根据前面的独立性假设，先分别计算，，等项，再将其结果相乘即可，而的计算公式为：
#
# 在实现算法时，需要考虑两个问题：
# a.当使用连乘计算时，若某一个词的概率为0，那么最终的结果也会为0，这是不正确的。为防止这种情况，需要将所有词项出现的次数都初始化为1，每一类所有词项数量初始化为2；
# b.在连乘时，为防止单项概率过小导致连乘结果下溢，需要对结果求自然对数将其转化为加法，因为。
# ⑤根据上一步计算出来的概率编写分类器函数。

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 其中，p0Vec，p1Vec，pClass1均为上一步函数的返回值，分别代表公式中的，以及。
# ⑥编写测试函数。

def trainNB1(trainMat, listClasses):
    listClasses_ = np.array(listClasses)
    trainMat_ = np.array(trainMat)

    pAb = np.sum(listClasses_) / len(listClasses)  # c1类的个数
    c0V = np.sum(trainMat_[np.where(listClasses_ == 0)], axis=0, keepdims=True)  # c0各个词频数
    c1V = np.sum(trainMat_[np.where(listClasses_ == 1)], axis=0, keepdims=True)  # c1各个词频数
    c0all = np.sum(c0V != 0) + 2  # c0类的总词数
    c1all = np.sum(c1V != 0) + 2  # c1类的总词数
    c0V += 1
    c1V += 1

    return c0V / c0all, c1V / c1all, pAb


def NB(X_train, X_test, y_train, y_test):
    myVocabList = createVocabList(X_train)

    trainMat = []  # 训练数据的BOW
    for postinDoc in X_train:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB1(trainMat, y_train)

    pred = []
    for i in X_test:
        thisDoc = setOfWords2Vec(myVocabList, i)
        pred.append(classifyNB(thisDoc, p0V, p1V, pAb))

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred)
    
    return precision, recall, f1

    # 测试
    # thisDoc = setOfWords2Vec(myVocabList, '通知 泰山 启林 的 童鞋 也 能 微信 预付 电费 啦'.split())
    # print(classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    data_model = DataModel()
    results = []
    for X_train, X_test, y_train, y_test in data_model.get_fold():
        results.append(NB(X_train, X_test, y_train, y_test))
    for _, (precision, recall, f1) in enumerate(results):
        print('fold: {}'.format(_+1))
        print('p: {}, r: {}, f1: {}'.format(precision, recall, f1))
