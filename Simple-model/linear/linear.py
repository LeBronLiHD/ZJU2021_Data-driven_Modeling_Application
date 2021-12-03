#定义线性回归模型
def test_LinearRegrssion():
    import parameters
    import load_data
    from sklearn import linear_model
    import pandas as pd 
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
    print(x_train.shape)
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)#训练数据
    print('Coefficients:%s,intercept %.2f' % (regr.coef_, regr.intercept_))#权重向量即为每个特征的相关系数
    y_test=regr.predict(x_test)
    data = pd.DataFrame(y_test)
    writer = pd.ExcelWriter('test.xlsx')
    data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
test_LinearRegrssion()