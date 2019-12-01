import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


# 创建数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 采用K近邻算法分类
def classifyByKNN(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 从文件中读取数据集
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试算法，计算算法错误率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classifyByKNN(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("The classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("The total error rate is :%f" % (errorCount / float(numTestVecs)))
# group, labels = file2matrix("datingTestSet.txt")
# group, ranges, minVals = autoNorm(group)
# result = classifyByKNN([0, 0, 0], group, labels, 3)
# print(result)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(group[:, 1], group[:, 2])
# plt.show()
# datingClassTest()