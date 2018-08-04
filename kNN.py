from numpy import *
import operator
from os import listdir
def classify0(inX, dataSet, labels, k):#用于分类的输入向量inx
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 将inX重复成dataSetSize行1列，tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)# sum(axis=1)就是将矩阵的每一行向量相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # argsort()得到的是排序后数据原来位置的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 确定前k个距离最小元素所在的主要分类labels
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 计算各个元素标签的出现次数(频率)，当voteIlabel在classCount中时，classCount.get()返回1，否则返回0
        # operator.itemgetter(1)表示按照第二个元素的次序对字典进行排序，reverse=True表示为逆序排序，即从大到小排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 最后返回发生频率最高的元素标签
def autoNorm(dataSet):#归一化操作
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals#最大值减去最小值
    normDataSet=zeros(shape(dataSet))#
    m=dataSet.shape(0)
    normDataSet=dataSet-tile(minVals,(m,1))#让dataset的每一个单数据减去重复了m行列数不变的列
    normDataSet=normDataSet/tile(ranges,(m,1))#上面的normDataSet除以ranges即可了。
    return normDataSet,ranges,minVals

