# coding:utf-8

from math import log
import operator
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
#
# import numpy as np


# # ①创建一个简单的数据集。这个数据集根据两个属性来判断一个海洋生物是否属于鱼类，第一个属性是不浮出水面是否可以生存，第二个属性是是否有鳍。数据集中的第三列是分类结果。
# def loadDataSet():
#     # 获取训练集
#     def get_iris_data():
#         iris = load_iris()
#         iris_data = iris.data
#         iris_target = iris.target
#         return iris_data, iris_target
#
#     iris_data, iris_target = get_iris_data()
#     # 将数据离散化
#     enc = OneHotEncoder()
#     iris_data = enc.fit_transform(iris_data).toarray().astype(np.int32)
#     train_x, test_x, train_y, test_y = train_test_split(iris_data, iris_target, test_size=0.33)
#     train_y = train_y.reshape((-1, 1))
#     test_y = test_y.reshape((-1, 1))
#     # print(train_x.shape)
#     # print(train_y.shape)
#     # print(train_x.dtype)
#     # print(train_y.dtype)
#     train = np.concatenate([train_x, train_y], axis=1)
#     test = np.concatenate([test_x, test_y], axis=1)
#     return train, test


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    # 这里的labels保存的是属性名称。
    labels = ['no surfacing', 'flippers']
    # dataSet = train_data
    return dataSet, labels


# ②编写函数计算熵。计算公式看前面。
def calcEntropy(dataSet):
    # 获取总的训练数据数
    numEntries = len(dataSet)
    # 创建一个字典统计各个类别的数据量
    labelCounts = {}
    for featVec in dataSet:
        # 使用下标-1获取所属分类保存到currentLabel
        # 【代码待补全】
        currentLabel = featVec[-1]
        # 若获得的类别属于新类别，则初始化该类的数据条数为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0
    for key in labelCounts.keys():
        # 计算p(xi)
        # 【代码待补全】
        p_xi = labelCounts[key] / numEntries
        # 计算熵
        # 【代码待补全】
        entropy -= p_xi * log(p_xi, 2)
    return entropy


# ③编写函数，实现按照给定特征划分数据集。

def splitDataSet(dataSet, axis, value):
    returnDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            # 隔开axis这一列提取其它列的数据
            # 保存到变量reducedFeatVec中
            # 【代码待补全】
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            returnDataSet.append(reducedFeatVec)
    return returnDataSet


# 	④实现特征选择函数。遍历整个数据集，循环计算熵和splitDataSet()函数，找到最好的特征划分方式。
def chooseBestFeatureToSplit(dataSet):
    # 获取属性个数，保存到变量numFeatures
    # 注意数据集中最后一列是分类结果
    # 【代码待补全】
    numFeatures = len(dataSet[0]) - 1

    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取数据集中某一属性的所有取值
        featList = [example[i] for example in dataSet]
        # 获取该属性所有不重复的取值，保存到uniqueVals中
        # 可使用set()函数去重
        # 【代码待补全】
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算按照第i列的某一个值分割数据集后的熵
            # 参考文档开始部分介绍的公式
            # 【代码待补全】
            newEntropy += calcEntropy(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# ⑤决策树创建过程中会采用递归的原则处理数据集。递归的终止条件为：程序遍历完所有划分数据集的属性；或者每一个分支下的所有实例都具有相同的分类。如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，在这种情况下，通常会采用多数表决的方法决定分类。

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# ⑥创建决策树。
def createTree(dataSet, labels):
    # 获取类别列表，类别信息在数据集中的最后一列
    # 使用变量classList
    # 【代码待补全】
    classList = [example[-1] for example in dataSet]
    # 以下两段是递归终止条件

    # 如果数据集中所有数据都属于同一类则停止划分
    # 可以使用classList.count(XXX)函数获得XXX的个数，
    # 然后那这个数和classList的长度进行比较，相等则说明
    # 所有数据都属于同一类，返回该类别即可
    # 【代码待补全】
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果已经遍历完所有属性则进行投票，调用上一步的函数
    # 注意，按照所有属性分割完数据集后，数据集中会只剩下
    # 一列，这一列是分类结果
    # 【代码待补全】
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 调用特征选择函数选择最佳分割属性，保存到bestFeat
    # 根据bestFeat获取属性名称，保存到bestFeatLabel中
    # 【代码待补全】
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 初始化决策树，可以先把第一个属性填好
    myTree = {bestFeatLabel: {}}
    # 删除最佳分离属性的名称以便递归调用
    del (labels[bestFeat])
    # 获取最佳分离属性的所有不重复的取值保存到uniqueVals
    # 【代码待补全】
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制属性名称，以便递归调用
        subLabel = labels[:]
        # 递归调用本函数生成决策树
        myTree[bestFeatLabel][value] = createTree \
            (splitDataSet(dataSet, bestFeat, value), subLabel)
    return myTree


def classify(inputTree, featLabels, testVec):
    # 获取树的第一个节点，即属性名称
    firstStr = list(inputTree.keys())[0]
    # 获取该节点下的值
    secondDict = inputTree[firstStr]
    # 获取该属性名称在原属性名称列表中的下标
    # 保存到变量featIndex中
    # 可使用index(XXX)函数获得XXX的下标
    # 【代码待补全】
    featIndex = featLabels.index(firstStr)
    # 获取待分类数据中该属性的取值，然后在secondDict
    # 中寻找对应的项的取值
    # 如果得到的是一个字典型的数据，说明在该分支下还
    # 需要进一步比较，因此进行循环调用分类函数；
    # 如果得到的不是字典型数据，说明得到了分类结果
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, featLabels = createDataSet()
    inputTree = createTree(dataSet, featLabels)
    dataSet, featLabels = createDataSet()
    testVec = [1, 0]
    print('input: ', testVec)
    print(classify(inputTree, featLabels, testVec))
