# -*- coding: utf-8 -*-

"""
data pre-process
1. data cleaning
    1. missing value
        1. delete the piece of data
        2. interpolation of value
            1. replace
            2. nearest neighbor imputation
            3. regression method
            4. spline interpolation
    2. outliers
        1. simple statistical analysis
            1. observe the maximum and minimum values and determine whether it is reasonable
            2. three delta in normal distribution
            3. box plot analysis
                1. upper quartile and lower quartile
                2. overcome the problem that delta in distribution is under the influence of outliers
    3. duplicated data
        1. analysis first, and remove it if the duplicated data makes no sense
    3. inconsistent data
2. data transformation
    1. square, square root, exponent, logarithm, etc.
    2. normalization
        1. maximum and minimum normalization
        2. zero mean normalization
    3. discretization of continuous data
    4. attribute structure, like BMI
"""

import load_data
import numpy as np
import parameters


def fill_nan_with_zero(data):
    count = 0
    shape_len = data.shape.__len__()
    for i in range(len(data)):
        if shape_len == 1:
            if np.isnan(data[i]):
                data[i] = 0
                count += 1
        else:
            for j in range(len(data[i])):
                if np.isnan(data[i][j]):
                    data[i][j] = 0
                    count += 1
    print("nan count ->", count)
    return data

def find_nan(data):
    #找到每一个特征的缺失值坐标
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    g=np.zeros([7,16], dtype = int)
    f=np.zeros([16], dtype = int)
    shape_len = data.shape.__len__()
    for i in range(len(data)):
        if shape_len == 1:
            if np.isnan(data[i]):
                f[count]=i
                count += 1
            
        else:
            for j in range(len(data[i])):
                if np.isnan(data[i][j]):
                    if j==0:
                        g[j,count0]=i
                        count0=count0+1
                    elif j==1:
                        g[j,count1]=i
                        count1=count1+1   
                    elif j==2:
                        g[j,count2]=i
                        count2=count2+1                        
                    elif j==3:
                        g[j,count3]=i
                        count3=count3+1                       
                    elif j==4:
                        g[j,count4]=i
                        count4=count4+1                        
                    elif j==5:
                        g[j,count5]=i
                        count5=count5+1                        
                    elif j==6:
                        g[j,count6]=i
                        count6=count6+1                        
                    else: 
                        g[j,count7]=i
                        count7=count7+1                
                
    if shape_len == 1:
        #print("nan count ->", count)
        return f,count
    else:
        '''''
        print("nan count0 ->", count0)
        print("nan count1 ->", count1)
        print("nan count2 ->", count2)
        print("nan count3 ->", count3)
        print("nan count4 ->", count4)
        print("nan count5 ->", count5)
        print("nan count6 ->", count6)
        '''
        a=np.array([count0, count1,count2, count3, count4, count5, count6])
        #print(a)
        return g,a

def data_cleaning(data):
    import matplotlib.pyplot as plt
    #进行样条插值
    import scipy.interpolate as spi
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


    return data

def pca_data(data):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(data)
    PCA(copy=True, n_components=None, whiten=False)
    print(pca.components_)  # 返回模型的各个特征向量
    print("*" * 50)
    print(pca.explained_variance_ratio_)  # 返回各个成分个字的方差百分比
    return pca.explained_variance_ratio_


if __name__ == '__main__':
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
    x_trainnan,  x_testnan = find_nan(x_train), find_nan(x_test)
    x_train, y_train, x_test = fill_nan_with_zero(x_train), \
                               fill_nan_with_zero(y_train), \
                               fill_nan_with_zero(x_test)
