import numpy as np


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("****original matrix******")
    print(np.shape(myMat))
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    print(numSV)
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    print(np.shape(reconMat))
    printMat(reconMat, thresh)

imgCompress(2)
