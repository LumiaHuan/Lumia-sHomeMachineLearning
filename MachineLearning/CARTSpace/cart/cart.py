from numpy import *

def loadData(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineSplitted = line.strip().split('\t')
            dataMat.append(map(float, lineSplitted))
    return dataMat


def binSplitDataSet(dataMatrix, feature, value):
    filterLt = (dataMatrix[:, feature]<value).A.flatten()
    matLt = dataMatrix[filterLt]
    filterGt = (dataMatrix[:, feature]>=value).A.flatten()
    matGt = dataMatrix[filterGt]
    return matLt, matGt


def regLeaf(dataSet, lam):
    return mean(dataSet[:,-1])

def regErr(dataSet, lam):
    return var(dataSet[:,-1]) * (shape(dataSet)[0])

def chooseBestSplit(dataMatrix, lam,leafType = regLeaf, errType=regErr, ops=(1,4)):
    if len(set(dataMatrix[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataMatrix, lam)
    tolS = ops[0];tolN=ops[1]
    bestIndex=0; bestValue=0
    bestS = inf
    S = errType(dataMatrix,lam)
    m,n = shape(dataMatrix)
    for featureIndex in xrange(n-1):
        for splitVal in set(dataMatrix[:,featureIndex].T.tolist()[0]):
            matLeft, matRight = binSplitDataSet(dataMatrix, featureIndex, splitVal)
            if (shape(matLeft)[0] < tolN) or (shape(matRight)[0] < tolN):
                continue
            errS = errType(matLeft,lam) + errType(matRight,lam)
            if errS < bestS:
                bestS = errS
                bestIndex = featureIndex
                bestValue = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataMatrix,lam)
    matLeft, matRight = binSplitDataSet(dataMatrix, bestIndex, bestValue)
    if (shape(matLeft)[0] < tolN) or (shape(matRight)[0] < tolN):
        return None, leafType(dataMatrix,lam)
    return bestIndex, bestValue

def createTree(dataMatrix, lam, leafType=regLeaf, errType=regErr, ops=(1,4) ):
    feat, val = chooseBestSplit(dataMatrix, lam,leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    ltMatrix, gtMatrix = binSplitDataSet(dataMatrix, feat, val)
    retTree['left'] = createTree(ltMatrix,lam, leafType, errType, ops)
    retTree['right'] = createTree(gtMatrix, lam,leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0




def linearSolver(dataMatrix, lam):
    m,n = shape(dataMatrix)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataMatrix[:,0:n-1]; Y = dataMatrix[:,-1]
    xTx = X.T * X
    xTx_lam = xTx + lam * mat(eye(shape(xTx)[0]))
    if linalg.det(xTx_lam) == 0.0:
        raise NameError('This matrix is singular!!')
    ws = xTx_lam.I * X.T * Y
    return ws, X, Y

def modelLeaf(dataMatrix, lam):
    ws,X,Y = linearSolver(dataMatrix,lam)
    return ws

def modelErr(dataMatrix, lam):
    ws, X, Y = linearSolver(dataMatrix,lam)
    yHat = X * ws
    return sum(power(Y-yHat, 2))


def regTreeEval(avgValue, inData):
    return float(avgValue)

def modelTreeEval(ws, inData):
    m,n = shape(inData)
    X = mat(ones((m,n+1)))
    X[:, 1:n+1] = inData
    return X*ws

def prune(tree, testData, treeEval, lam):
    if shape(testData)[0] == 0: print 'test is none';return getMean(tree)
    if (isTree(tree['left'])) or (isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet,treeEval,lam)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet,treeEval,lam)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        treeEvalLeftValues = treeEval(tree['left'], lSet[:,0:-1])
        treeEvalRightValues = treeEval(tree['right'], rSet[:,0:-1])
        errorNoMerge = sum(power(lSet[:,-1]-treeEvalLeftValues,2))+\
            sum(power(rSet[:,-1]-treeEvalRightValues,2))

        if treeEval.__name__ == "regTreeEval":
            treeMean = (treeEvalLeftValues + treeEvalRightValues) / 2.0
            errorMerge = sum(power(testData[:,-1] - treeMean,2))
            if errorMerge < errorNoMerge:
                print 'merge'
                return treeMean
            else: return tree
        elif treeEval.__name__ == "modelTreeEval":
            ws, X, Y = linearSolver(testData,lam)
            yHat = X * ws
            errorMerge = sum(power(Y-yHat, 2))
            if errorMerge < errorNoMerge:
                print 'merge'
                return ws
            else: return tree
    else:
        return tree

def treeForecast(tree, inData, treeEval = regTreeEval):
    if not isTree(tree):
        return treeEval(tree, inData)
    if inData[tree['spInd']] < tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, treeEval)
        else:
            return treeEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, treeEval)
        else:
            return treeEval(tree['right'], inData)


def createForecast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in xrange(m):
        yHat[i] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat

def calcRss(tree, dataMatrixWithLabel, modelEval):
    realLabel = dataMatrixWithLabel[:,-1]
    predict = createForecast(tree, dataMatrixWithLabel[:,0:-1], modelEval)
    diff = realLabel - predict
    diff_square = power(diff, 2)
    return sum(diff_square)