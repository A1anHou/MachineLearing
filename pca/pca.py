import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    print(np.shape(dataMat))
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVects)
    eigValInd = np.argsort(eigVals)
    print(eigValInd)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    print(eigValInd)
    redEigVects = eigVects[:, eigValInd]
    print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
plt.show()
