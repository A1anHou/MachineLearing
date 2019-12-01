import numpy as np
import kNN
from os import listdir


# 将32*32的图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    # 循环读出文件前32行
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 测试代码
def handwritingClassTest():
    hwLabels = []
    trainFileList = listdir('trainingDigits')
    m = len(trainFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = kNN.classifyByKNN(vectorUnderTest, trainingMat, hwLabels, 3)
        print("The classifier came back with :%d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1
    print("\nThe total number of errors is: %d" % errorCount)
    print("\nThe total error rate is :%f" % (errorCount / float(mTest)))

handwritingClassTest()
