import numpy as np

# 从文本中构建矩阵，加载文本文件，然后处理
def loadDataSet(fileName):    # 通用函数，用来解析以 tab 键分隔的 floats（浮点数），例如: 1.658985    4.285136
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)    # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat
# 计算两个向量的欧式距离（可根据场景选择）
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2))) # la.norm(vecA-vecB)
# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，
# 这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，
# 以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = np.shape(dataSet)[1] # 列的数量
    centroids = np.mat(np.zeros((k,n))) # 创建k个质心矩阵
    for j in range(n): # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j])    # 最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)    # 范围 = 最大值 - 最小值
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))    # 随机生成
    return centroids
# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，直到数据点的簇分配结果不再改变位置。
# 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的,因为数据足够相似，也可能会陷入局部最小值）
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]    # 行数
    clusterAssment = np.mat(np.zeros((m, 2))) # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataSet, k)  # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 循环每一个数据点并分配到最近的质心中去
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                clusterAssment[i, :] = minIndex,minDist**2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print (centroids)
        for cent in range(k): # 更新质心
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取该簇中的所有点
            centroids[cent,:] = np.mean(ptsInClust, axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment
