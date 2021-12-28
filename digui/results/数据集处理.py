import copy
import numpy as np
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# datacleaning()

# 读取数据
a = np.loadtxt('train_input.csv', dtype=np.float64)
b = np.loadtxt('output.txt', dtype=np.float64)
b = b[:, np.newaxis]
# 按行拼接，训练集输出也是信息
values = np.concatenate((a, b), axis=1)
# 深复制而非浅复制，很有意思的
data = copy.deepcopy(values)
data = pd.DataFrame(data)


def series_to_supervised(data, n_in, n_out):
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
    # 删除缺失值
    agg.dropna(inplace=True)
    return agg


# 确保所有数据是 float64 类型
data = data.astype('float64')
data