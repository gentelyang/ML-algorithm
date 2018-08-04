from math import  log
import  operator
#优点：复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据；缺点是：产生过度匹配问题
#首先计算shannonEnt
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)#数据集的长度
    labelCounts={}#创建一个字典，用于存放标签和数据，key为标签
    for featVec in dataSet:#遍历数据
        currentLabel=featVec[-1]#数据的最后一列为标签
        if currentLabel not in labelCounts.keys():#如果currentLabel（当前标签）不存在字典中，则字典中此标签为0
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1#如果在字典中，则将此标签的个数值加一，labelcounts中的value代表的是此标签的数据出现的次数
        shnnonEnt=0.0#初始化shennonEnt
        for key in labelCounts:#当遍历完所有的数据，打上标签和计算出次数后，接下来开始计算shannonEnt的值
            prob=float(labelCounts[key])/numEntries#先计算概率值，计算每一个标签的概率值
            shannonEnt=-prob*log(prob,2)#利用计算的每一个key标签的概率去计算shannonEnt
        return shannonEnt
#接下来开始划分数据集
def splitDataSet(dataSet, axis, value):
    """
    Function：   按照给定特征划分数据集

    Args：       dataSet：带划分的数据集
                axis：划分数据集的特征或属性
                value：需要返回的特征的值

    Returns：    retDataSet：符合特征的数据集
    """
    #创建新的list对象
    retDataSet = []
    #抽取数据集
    for featVec in dataSet:
        #将符合特征的数据抽取出来，axis=0，1，2，3..，value为对应的可能
        if featVec[axis] == value:
            #截取列表中第axis个之前的数据，这里axis就是特征，因为axis特征要用于划分属性，所以将它截取出来
            reducedFeatVec = featVec[:axis]
            #将第axis+1之后的数据接入到上述数据集
            reducedFeatVec.extend(featVec[axis+1:])
            #将处理结果作为列表接入到返回数据集
            retDataSet.append(reducedFeatVec)
    #返回符合特征的数据集,去除axis这个属性的值，就是特征相同的数据集放在一块，[[特征1],[特征2】，..]，形如这个样子。
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    """
    Function：   选择最好的数据集划分方式
    Args：       dataSet：待划分的数据集
    Returns：    bestFeature：划分数据集最好的特征
    """
    #初始化特征数量，dataSet[0]指的是一列，len(dataSet[0])指的是列的长度，即属性的个数。
    numFeatures = len(dataSet[0]) - 1
    #计算原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #初始化信息增益和最佳特征
    bestInfoGain = 0.0; bestFeature = -1
    #选出最好的划分数据集的特征
    for i in range(numFeatures):
        #创建唯一的分类标签列表，找每一个特征为i时的数据时都去遍历一遍数据集。
        #语句featList = [example[i] for example in dataSet]作用为：
#将dataSet中的数据按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
        #语句classList = [example[-1] for example in dataSet]作用为：
#将dataSet中的数据按行依次放入example中，然后取得example中的example[-1]元素，放入列表classList中
        #list_0 = [x*x for x in range(5)]  print(list_0)
        featList = [example[i] for example in dataSet]
        #从列表中创建集合，以得到列表中唯一元素值
        uniqueVals = set(featList)
        #初始化香农熵
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            #调用前面的dplitDataSet函数，将数据集按照特征i和对应的value进行划分，并将subDataSet作为返回值。
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))#计算每一类的概率
            newEntropy += prob * calcShannonEnt(subDataSet)#计算每一类的shannoEnt值
        #得到信息增益，Ent(D)-(Dv/D)*Ent(Dv)
        infoGain = baseEntropy - newEntropy
        #计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    #返回最好的特征
    return bestFeature
def majorityCnt(classList):
    """
     如果数据集已经处理了所有的属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，书中采用的是多数表决的方法决定该叶子节点的分
    Function：   决定叶子结点的分类
    Args：       classList：分类列表
    Returns：    sortedClassCount[0][0]：叶子结点分类结果
    """
    #创建字典
    classCount={}
    #给字典赋值
    for vote in classList:
        #如果字典中没有该键值，则创建
        if vote not in classCount.keys():
            classCount[vote] = 0
        #为每个键值计数
        classCount[vote] += 1
    #对classCount进行排序，reverse=True为逆序排序。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回叶子结点分类结果
    return sortedClassCount[0][0]
def createTree(dataSet, labels):
    """
    Function：   创建树
    Args：       dataSet：数据集
                labels：标签列表
    Returns：    myTree：创建的树的信息
    """
    #创建分类列表
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选取最好的分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #创建字典存储树的信息
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    #从列表中创建集合
    uniqueVals = set(featValues)
    #遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        #复制类标签
        subLabels =labels[:]
        #递归调用函数createTree()，返回值将被插入到字典变量myTree中
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #返回字典变量myTree
    return myTree