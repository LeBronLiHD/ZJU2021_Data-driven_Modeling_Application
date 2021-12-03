# 加载数据集
import numpy as np
import preprocess
a=np.loadtxt('./data/Btrain_input.txt',dtype=np.float32)
c=np.loadtxt('./data/Btest_input.txt',dtype=np.float32)
'''''
x_trainnan, x_trainn= find_nan(a)
x_testnan ,x_testn=find_nan(c)
print(x_trainnan, x_trainn)
'''
a=preprocess.data_cleaning(a)
c=preprocess.data_cleaning(c)
a=preprocess.smaller_than_zero(a)
c=preprocess.smaller_than_zero(c)
np.savetxt('./data/train_input.txt',a)
np.savetxt('./data/test_input.txt',c)
preprocess.modifydata()
#fill_nan_with_zero(a)
#pca_data(a)