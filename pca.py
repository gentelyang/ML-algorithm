from numpy import  *
def loadDataSet(filename,delim='\t'):
    fr=open(filename)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[map(float,line) for line in stringArr]
    return map(datArr)
def pca(dataMat,topNfeat=9999999):
    meanvlas=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanvlas
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanvlas
    return lowDDataMat,reconMat