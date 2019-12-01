from math import log
import operator
import treePlotter


# 创建模拟数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 获取数据标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算标签为key的情况出现的概率
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 划分数据集，参数待划分的数据集，划分数据集的特征，需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 将特征值放入一个集合中，消除重复特征值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = bestEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 若数据集已处理完所有属性，但类标签仍然不唯一，则采用多数表决方式决定类标签
def majorityCht(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 所有样本属于同一类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 所有属性均处理完毕
    if len(dataSet[0]) == 1:
        return majorityCht(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classifyByTree(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = ''
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classifyByTree(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 存储决策时
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(myTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def createDataSetFromFile(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lensesLabels


myDat, labels = createDataSetFromFile("lenses.txt")
print(myDat)
# 切片复制不会受浅复制影响
mLabels = labels[:]
myTree = createTree(myDat, labels)
# result = classifyByTree(myTree, mLabels, [1, 0])
print(myTree)
# storeTree(myTree, "test.txt")
treePlotter.createPlot(myTree)
# print(myDat)
# print(splitDataSet(myDat, 0, 1))
# print(splitDataSet(myDat, 0, 0))
# print(chooseBestFeatureToSplit(myDat))
# print(calcShannonEnt(myDat))
