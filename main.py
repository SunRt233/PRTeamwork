from math import sqrt
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

def getData(path):
    df = pd.read_excel(path,'Data')
    X = df.iloc[:,6:28].values[1:]
    Y = df[['NSP']].values[1:]
    
    return [Y,X]

def pcaProcess(data,dmins=10):
    Y,X = data
    scalor = StandardScaler()
    
    pca = PCA(n_components=dmins)
    # pca = PCA()
    scaledX = scalor.fit_transform(X)
    X_pca = pca.fit_transform(scaledX)
    print('主成分方差贡献率：',pca.explained_variance_ratio_)
    # print('累计主成分方差贡献率：',np.cumsum(pca.explained_variance_ratio_))
    
    return [Y,X_pca]

def splitData(data,ratio=0.6):
    Y,X = data
    Y_train = Y[:int(ratio*len(Y))]
    X_train = X[:int(ratio*len(X))]
    Y_test = Y[int(ratio*len(Y))+1:]
    X_test = X[int(ratio*len(X))+1:]
    
    return [Y_train,X_train,Y_test,X_test]

def knnProcess(data):
    Y_train,X_train = data
    k = int(min(35,sqrt(len(Y_train))))
    print('K:',k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    
    return knn

def main():
    while input('继续运行？[Y/N]') == 'Y':
        dmins = int(input('输入PCA维度:'))
        ratio = float(input('输入训练集比例:'))
        print('PCA维度:',dmins,'训练集比例:',ratio)
        
        data = getData(r'./数据及说明/S3-CTG.xlsx')
        Y_train,X_train,Y_test,X_test = splitData(data,ratio)
        
        data_pca = pcaProcess([Y_train,X_train],dmins)
        knn = knnProcess(data_pca)
        
        Y_test_pca,X_test_pca = pcaProcess([Y_test,X_test],dmins)
        result = knn.predict(X_test_pca)
        
        print(len(X_test_pca) == len(result))
        
        correct_num = 0
        total_num = len(result)
        for i in range(0,len(result)-1):
            # print(i,result[i],Y_test_pca[i])
            if result[i] == Y_test_pca[i]:
                correct_num += 1
        print('正确率:',correct_num/total_num)
    
main()
    
    
    
    
