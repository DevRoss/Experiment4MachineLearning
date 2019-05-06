# coding: utf-8

# -*- coding:utf-8 -*-
from numpy import *
import os


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def kNNClassify(newInput, dataSet, labels, k):
    [N, M] = dataSet.shape
    # calculate the distance between testX and other training samples
    difference = tile(newInput, (N, 1)) - dataSet  # tile for array and repeat for matrix in Python, == repmat in Matlab
    difference = difference ** 2  # take pow(difference,2)
    distance = difference.sum(1)  # take the sum of difference from all dimensions，得到于训练集中每个样本的距离
    distance = distance ** 0.5
    sortedDistIndices = distance.argsort()

    # find the k nearest neighbours
    classCount = {}  # create the dictionary
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,
                                               0) + 1  # get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
    predictedClass = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
    return predictedClass[0][0]


def img2vector(filename):
    fileIn = open(filename)
    returnVec = list()
    for i in range(32):
        # 读取整行数据
        lineStr = fileIn.readline()
        for j in range(32):
            returnVec.append(lineStr[j])
    fileIn.close()
    return returnVec


def loadDataSet():
    ## Step1: 获取训练集
    print("---Getting training set...")
    # 根据实际情况设置数据集路径
    dataSetDir = ""
    trainingFileList = os.listdir(dataSetDir + "trainingDigits")
    numSamples = len(trainingFileList)

    train_x = zeros((numSamples, 1024))
    train_y = []  # 保存标签数据，即分类结果
    for i in range(numSamples):
        filename = trainingFileList[i]
        train_x[i, :] = img2vector(dataSetDir + "trainingDigits/%s" % filename)
        label = int(filename.split('_')[0])
        train_y.append(label)

    ## Step2: 获取测试集
    print("---Getting test set...")
    testFileList = os.listdir(dataSetDir + "testDigits")
    numSamples = len(testFileList)

    test_x = zeros((numSamples, 1024))
    test_y = []  # 保存标签数据，即分类结果
    for i in range(numSamples):
        filename = testFileList[i]
        test_x[i, :] = img2vector(dataSetDir + "testDigits/%s" % filename)
        label = int(filename.split('_')[0])
        test_y.append(label)
    return train_x, train_y, test_x, test_y


def testHandWritingClass():
    # 获取数据集
    print("Step 1: Load data...")
    # 要求使用train_x, train_y, test_x, test_y作为变量
    train_x, train_y, test_x, test_y = loadDataSet()

    # 由于knn是不需要训练步骤的，所以这里直接使用pass跳过
    print("Step 2: Training...")
    pass

    print("Step 3: Testing and Showing the result...")
    numTestSamples = test_x.shape[0]
    # 对测试集中的数据进行分类，取k=2,3,4,5，将得到的结果与标签对
    # 比，如果相等则分类正确的数目加一
    for k in range(2, 6):
        matchCount = 0  # 用以统计分类正确的数目
        for i in range(numTestSamples):
            cls = kNNClassify(test_x[i], train_x, train_y, k)
            if cls == test_y[i]:
                matchCount += 1
        # 所有测试集数据都跑完后计算分类的正确率，保存到acuuracy变量
        accuracy = matchCount / numTestSamples
        print("When k is", k, "the classify accuracy is %.2f%%" % (accuracy * 100))

    # 显示结束
    print("Step 4: Finish...")


testHandWritingClass()
