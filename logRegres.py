import  numpy as np
import  matplotlib.pyplot as plt
#加载数据
def loadDataSet():
    dataMat=[];labelMat=[]
    path=r'指定的文件路径'
    fr=open(path)
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMatrix=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones(n,1)
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)#所有样本点更新回归系数
        error=(labelMatrix-h)
        weight=weights+alpha*dataMatrix.transpose()*error
#画出决策边界，这个就不写了。
def stocGradAscent(dataMatrix,classLabels):
    dataMatrix=np.array(dataMatrix)
    m,n=np.shape(dataMatrix)
    alpha=0.1
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))#对单独一个样本更新回归系数
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights