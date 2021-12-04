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

import numpy as np
def smaller_than_zero(data):
    count = 0
    shape_len = data.shape.__len__()
    for i in range(len(data)):
        if shape_len == 1:
            if data[i]<=0:
                data[i] = data[i-1]+data[i+1]
                count += 1
        else:
            for j in range(len(data[i])):
                if data[i][j]<0:
                    data[i][j] = data[i-1][j]+data[i+1][j]
                    count += 1
    print("nan count ->", count)
    return data


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
    count8 = 0
    count9 = 0
    count10 = 0
    count11= 0
    count12= 0
    count13= 0
    count14= 0
    count15= 0
    count16 = 0
    count17 = 0
    count18 = 0
    count19 = 0
    count20 = 0
    count21= 0
    count22= 0
    count23= 0
    count24= 0
    count25= 0
    g=np.zeros([26,50], dtype = int)
    f=np.zeros([50], dtype = int)
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
                    elif j==7:
                        g[j,count7]=i
                        count7=count7+1   
                    elif j==8:
                        g[j,count8]=i
                        count8=count8+1                        
                    elif j==9:
                        g[j,count9]=i
                        count9=count9+1                       
                    elif j==10:
                        g[j,count10]=i
                        count10=count10+1
                    elif j==11:
                        g[j,count11]=i
                        count11=count11+1   
                    elif j==12:
                        g[j,count12]=i
                        count12=count12+1                        
                    elif j==13:
                        g[j,count13]=i
                        count13=count13+1                       
                    elif j==14:
                        g[j,count14]=i
                        count14=count14+1                        
                    elif j==15:
                        g[j,count15]=i
                        count15=count15+1                        
                    elif j==16:
                        g[j,count16]=i
                        count16=count16+1 
                    elif j==17:
                        g[j,count17]=i
                        count7=count7+1   
                    elif j==18:
                        g[j,count18]=i
                        count18=count18+1                        
                    elif j==19:
                        g[j,count19]=i
                        count19=count19+1                       
                    elif j==20:
                        g[j,count20]=i
                        count20=count20+1
                    elif j==21:
                        g[j,count21]=i
                        count21=count21+1   
                    elif j==22:
                        g[j,count22]=i
                        count22=count22+1                        
                    elif j==23:
                        g[j,count23]=i
                        count23=count23+1                       
                    elif j==24:
                        g[j,count24]=i
                        count24=count24+1                        
                    else:
                        g[j,count25]=i
                        count25=count25+1                                                    
                               
                
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
        a=np.array([count0, count1,count2, count3, count4, count5, count6,count7, count8,count9, count10, count11,count12, count13, count14, count15, count16,count17, count18,count19,count20, count21,count22, count23, count24, count25])
        #print(a)
        return g,a

def find_strange(data):
    #找到每一个特征的零值坐标
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    g=np.zeros([7,50], dtype = int)
    f=np.zeros([50], dtype = int)
    shape_len = data.shape.__len__()
    for i in range(len(data)):
        if shape_len == 1:
            if (data[i]<=0)|(data[i]>=1):
                f[count]=i
                count += 1  
        else:
            for j in range(len(data[i])):
                if (data[i][j]<=0)|(data[i][j]>=1):
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
        return f,count
    else:
        a=np.array([count0, count1,count2, count3, count4, count5, count6])
        return g,a

def ployinterp_column(s,i,n):
    # 自定义列向量插值函数
    from scipy.interpolate import lagrange
    k = 3
    #print(s.shape)
    # 取前后k个数的索引
    a=s[i][np.arange(n-k,n)]
    b=s[i][np.arange(n+1,n+1+k)]
    #2k个点
    y=np.hstack((a,b))
    #y = y[y.notnull()] # 剔除空值
    #2k个点对应坐标
    x=np.hstack((np.arange(n-k,n),np.arange(n+1,n+1+k)))
    # 返回这2k个数据的拉格朗日多项式函数在n的值
    return lagrange(x,y)(n)

def ployinterp1_column(s,i,n,k):
    # 自定义插值函数
    from scipy.interpolate import lagrange
    #print(s.shape)
    # 取后k个数的索引
    y=s[i][np.arange(n+1,n+1+k)]
    #k个点对应坐标
    x=np.arange(n+1,n+1+k)
    # 返回这2k个数据的拉格朗日多项式函数在n的值
    return lagrange(x,y)(n)

def ployinterp2_column(s,i,n,k):
    # 自定义插值函数(针对最后一个值需要插值进行改变)
    from scipy.interpolate import lagrange
    #print(s.shape)
    # 取前k个数的索引
    y=s[i][np.arange(n-k,n)]
    #k个点对应坐标
    x=np.arange(n-k,n)
    # 返回这2k个数据的拉格朗日多项式函数在n的值
    return lagrange(x,y)(n)

def data_cleaning(data):
    # 过滤数据,将异常值设成None(没写)
    a, b= find_nan(data)
    #c, d= find_strange(data)
    data=np.swapaxes(data,0,1)
    #判断是否需要插值    
    for i in  range(len(b)):
        for j in range(b[i]):
            data[i,a[i][j]]=ployinterp_column(data,i,a[i][j])
    '''''
    for i in  range(len(d)):
        for j in range(d[i]):
            if i<1:
                data[i,c[i][j]]=ployinterp1_column(data,i,c[i][j],5)
            elif i<d[i]-5:
                data[i,c[i][j]]=ployinterp_column(data,i,c[i][j])
            else:
                data[i,c[i][j]]=ployinterp2_column(data,i,c[i][j],5)
    '''
    data=np.swapaxes(data,0,1)
    #find_strange(data)
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
def modifytraindata():
    # tn-2,tn-1,tn
    import preprocess
    import numpy as np
    a=np.loadtxt('./data/train_input.txt',dtype=np.float32)
    #data2删去前两行
    data1=np.delete(a, 0, axis = 0)
    data2=np.delete(data1, 0, axis = 0)
    #data3删去第一行和最后一行
    data3=np.delete(data1, len(data1)-1, axis = 0)
    #data4删去最后两行
    data4=np.delete(a, len(a)-1, axis = 0)
    data4=np.delete(data4, len(data4)-1, axis = 0)
    #按照data4、data3、data2排列
    data=np.concatenate((data4,data3),axis=1)
    data=np.concatenate((data,data2),axis=1)
    np.savetxt('./data/train_modify.txt',data)
    
def modifytestdata():
    import numpy as np
    a=np.loadtxt('./data/train_input.txt',dtype=np.float32)
    data1=a[1594:,:]
    print(data1.shape)
    b=np.loadtxt('./data/test_input.txt',dtype=np.float32)
    data0=np.concatenate((data1,b),axis=0)
    data1=np.delete(data0, len(data0)-1, axis = 0)
    data1=np.delete(data1, len(data1)-1, axis = 0)
    data2=np.delete(data0, len(data0)-1, axis = 0)
    data2=np.delete(data2, 0, axis = 0)
    data3=np.delete(data0, [0,1], axis = 0)
    data=np.concatenate((data1,data2,data3),axis=1)
    np.savetxt('./data/test_modify.txt',data)
def modifyoutput():
    a=np.loadtxt('./data/output.txt',dtype=np.float32)
    a=np.delete(a, [0,1], axis = 0)
    np.savetxt('./data/output_modify.txt',a)
def modifydata():
    modifytraindata()
    modifytestdata()
    modifyoutput()

if __name__ == '__main__':
    print("hh")
