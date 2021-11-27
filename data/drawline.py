from pandas import read_csv
from matplotlib import pyplot
import numpy as np
a=np.loadtxt('Btrain_input.txt',dtype=np.float32)
b=np.loadtxt('output.txt',dtype=np.float32)
print(a.shape)
print(b.shape)
k=0
for i in range(len(a)):   
    for j in range(len(a[i])):
        if np.isnan(a[i][j]):
            #print(i,j)
            a[i][j]=0
            k=k+1
for i in range(len(b)):   
        if np.isnan(b[i]):
            #print(i)
            b[i]=0
print(k)
np.save("x_train.npy",a)
np.save("y_train.npy",b)

#x_train=np.swapaxes(a,1,2)
# 指定要绘制的列
groups = [0, 1, 2, 3, 4, 5, 6]
i = 1
# 绘制每一列
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(a[:, group])
    pyplot.title(group+1, y=0.5, loc='right')
    i += 1
pyplot.show()