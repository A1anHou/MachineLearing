import numpy as np


# 创建模拟数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 建立词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 求两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 检测词汇是否出现在词汇表中，输入参数词汇表，文档
# 词集模型，每个词只能出现一次
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word : %s is not in my Vocabulary!" % word)
    return returnVec


# 词袋模型，每个词可以出现多次
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("The word : %s is not in my Vocabulary!" % word)
        return returnVec


# 训练朴素贝叶斯分类器,输入参数文档矩阵，文档类别标签向量
def trainNB0(trainMatrix, trainCategory):
    # 行数
    numTrainDocs = len(trainMatrix)
    # 列数
    numWords = len(trainMatrix[0])
    # 因为处理二分类问题，所以只用计算1类别出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 为防止概率连乘时出现概率为0的情况，将所有分子初始化为1，分母初始化为2
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 为防止小概率连乘导致下溢出，将概率取对数
    # 1类别中各个词出现的概率
    p1Vect = np.log(p1Num / p1Denom)
    # 0类别中各个次出现的概率
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 利用朴素贝叶斯分类器分类
def classifyByNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对应向量之间相乘
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmatian']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyByNB(thisDoc, p0V, p1V, pAb))


testingNB()
