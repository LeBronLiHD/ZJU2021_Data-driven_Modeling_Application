def trans(n):
    '''''
    使用示范
    import transform_shape
    transform_shape.trans(20)
    返回train_x、test_x，形状(1456, 987)、(789, 987)
    '''
    # 导入相关包
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    # 读取数据
    a=np.loadtxt('train_input.csv', dtype=np.float64)
    b=np.loadtxt('test_input.csv', dtype=np.float64)
    #print(pd.DataFrame(a).isnull().sum(axis=0))    #无缺失值
    print(a.shape)  # 1596条数据，7个特征
    print(b.shape)
    data=np.concatenate((a,b),axis=0)
    print(data.shape)
    # 查看数据分布
    #data.plot(subplots=True, figsize=(20, 10), legend = True)
    #data.head()
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        数据处理
        :param data:数据
        :param n_in:输入特征个数
        :param n_out:目标值
        :param dropnan:是否删除 Nan 值 
        :return:
        """
        df = pd.DataFrame(data)
        n_vars = df.shape[1]  # n_vars 列数
        cols, names = list(), list()
        
        # 时间间隔跨度, 时间点个数，共 n_in 个
        # 首先添加当前时刻之前的时间点
        for i in range(n_in - 1, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 然后添加当前时刻
        cols.append(df)
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]

        # 添加 target 为未来 n_out 分钟后时刻的温度
        cols.append(df.shift(-n_out))
        names += [('var%d(t+%d)' % (j + 1, n_out)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    dataset=series_to_supervised(data, n_in=n*7, n_out=1, dropnan=True)
    print(dataset.shape)
    #dataset.head()
    # 归一化特征
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    values=scaled_data
    train_x = values[:-798,:]
    print(train_x.shape)
    test_x = values[-798:,:]
    print(test_x.shape)
    return train_x, test_x