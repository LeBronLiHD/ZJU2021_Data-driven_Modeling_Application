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
import matplotlib.pyplot as plt


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


def expand_shape(data, expand=False, shuffle=False):
    # expand = False: (21, ) -> (21, 21)
    # expand = True:  (21, ) -> (42, 42)
    exp_data = []
    if expand == False:
        for i in range(parameters.G_NumOfFeature):
            row = []
            for j in range(parameters.G_NumOfFeature):
                row.append(data[j])
            row = np.array(row)
            exp_data.append(row)
    else:
        for i in range(parameters.G_NumOfFeature * 2):
            row = []
            for j in range(parameters.G_NumOfFeature * 2):
                row.append(data[int(j/2)])
            row = np.array(row)
            exp_data.append(row)
    return np.array(exp_data)


def transfer_to_image(data):
    for_show = []
    for k in range(np.shape(data)[0]):
        for_show_unit = []
        for j in range(np.shape(data)[1]):
            for_show_unit.append(data[k][j][0])
        for_show.append(for_show_unit)
    return for_show


def let_me_see_see(image):
    plt.figure()
    # for_show = transfer_to_image(image)
    plt.imshow(image)
    plt.ylabel("let_me_see_see")
    plt.show()


def transfer_x_y(x_train, show_image=False):
    exp_x_train = []
    for i in range(len(x_train)):
        exp_x_train.append(expand_shape(x_train[i], expand=True, shuffle=False))
    if show_image:
        let_me_see_see(exp_x_train[0])
    return np.array(exp_x_train)


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
    #???????????????????????????????????????
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    g=np.zeros([7,17], dtype = int)
    f=np.zeros([17], dtype = int)
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

def find_strange(data):
    #????????????????????????????????????
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
    # ??????????????????????????????
    from scipy.interpolate import lagrange
    k = 3
    #print(s.shape)
    # ?????????k???????????????
    a=s[i][np.arange(n-k,n)]
    b=s[i][np.arange(n+1,n+1+k)]
    #2k??????
    y=np.hstack((a,b))
    #y = y[y.notnull()] # ????????????
    #2k??????????????????
    x=np.hstack((np.arange(n-k,n),np.arange(n+1,n+1+k)))
    # ?????????2k??????????????????????????????????????????n??????
    return lagrange(x,y)(n)

def ployinterp1_column(s,i,n,k):
    # ?????????????????????
    from scipy.interpolate import lagrange
    #print(s.shape)
    # ??????k???????????????
    y=s[i][np.arange(n+1,n+1+k)]
    #k??????????????????
    x=np.arange(n+1,n+1+k)
    # ?????????2k??????????????????????????????????????????n??????
    return lagrange(x,y)(n)

def ployinterp2_column(s,i,n,k):
    # ?????????????????????(?????????????????????????????????????????????)
    from scipy.interpolate import lagrange
    #print(s.shape)
    # ??????k???????????????
    y=s[i][np.arange(n-k,n)]
    #k??????????????????
    x=np.arange(n-k,n)
    # ?????????2k??????????????????????????????????????????n??????
    return lagrange(x,y)(n)

def data_cleaning(data):
    # ????????????,??????????????????None(??????)
    a, b= find_nan(data)
    c, d= find_strange(data)
    data=np.swapaxes(data,0,1)
    #????????????????????????    
    for i in  range(len(b)):
        for j in range(b[i]):
            data[i,a[i][j]]=ployinterp_column(data,i,a[i][j])

    for i in  range(len(d)):
        for j in range(d[i]):
            if i<1:
                data[i,c[i][j]]=ployinterp1_column(data,i,c[i][j],5)
            elif i<d[i]-5:
                data[i,c[i][j]]=ployinterp_column(data,i,c[i][j])
            else:
                data[i,c[i][j]]=ployinterp2_column(data,i,c[i][j],5)
    data=np.swapaxes(data,0,1)
    find_strange(data)
    return data

def pca_data(data):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(data)
    PCA(copy=True, n_components=None, whiten=False)
    print(pca.components_)  # ?????????????????????????????????
    print("*" * 50)
    print(pca.explained_variance_ratio_)  # ??????????????????????????????????????????
    return pca.explained_variance_ratio_
def modifytraindata():
    # tn-2,tn-1,tn
    import preprocess
    import numpy as np
    a=np.loadtxt('./data/train_input.txt',dtype=np.float32)
    #data2???????????????
    data1=np.delete(a, 0, axis = 0)
    data2=np.delete(data1, 0, axis = 0)
    #data3??????????????????????????????
    data3=np.delete(data1, len(data1)-1, axis = 0)
    #data4??????????????????
    data4=np.delete(a, len(a)-1, axis = 0)
    data4=np.delete(data4, len(data4)-1, axis = 0)
    #??????data4???data3???data2??????
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
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
    x_trainnan,  x_testnan = find_nan(x_train), find_nan(x_test)
    x_train, y_train, x_test = fill_nan_with_zero(x_train), \
                               fill_nan_with_zero(y_train), \
                               fill_nan_with_zero(x_test)
    a=np.loadtxt('./data/Btrain_input.txt',dtype=np.float32)
    b=np.loadtxt('./data/Btrain_output.txt',dtype=np.float32)
    c=np.loadtxt('./data/Btest_input.txt',dtype=np.float32)
    a=data_cleaning(a)
    c=data_cleaning(c)
    np.savetxt('./data/train_input.txt',a)
    np.savetxt('./data/test_input.txt',c)

