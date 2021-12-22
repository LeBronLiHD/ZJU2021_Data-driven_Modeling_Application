import numpy as np
import pandas as pd
import os
import math
import copy
import tensorflow.keras.backend as K 
from sklearn import preprocessing
from tensorflow.keras.models import load_model
import joblib

# 加载模型
model_path = 'results/lstm.h5'
model = load_model(model_path)

def predict(sequence):
    '''
    对输入序列进行预测，sequence为一360分钟的时序数据，
    输出结果为该序列第 60 分钟后的预测温度值
    param: sequence: np.array矩阵 shape:[360,6] [时间步, 特征]，
                     其中特征的索引顺序与数据集相同，分别为:
                     [outdoor_temperature,outdoor_humidity,indoor_temperature,
                      indoor_humidity,fan_status,power_consumption]
    return: 温度（标量），浮点数表示，限定使用np.float64或者python的float类型
    '''
    # 数据处理
    sequence = sequence.astype('float64')
    a = np.roll(sequence,4)#向后移位
    for i in range(6):
        a[i]=0
    sequence = sequence-a
    sequence = np.delete(sequence, [0,1,2,3], 0) # 删除前4行
    scaler = joblib.load('results/scaler.save')
    sequence = scaler.transform(sequence)
    sequence = np.expand_dims(sequence, 0).repeat(1, axis=0)
    pre_temperature = model.predict(sequence)
    
    # scaler只能对整体反归一化，构造两个空矩阵
    temp_array_1 = np.ones((len(sequence),2))
    temp_array_2 = np.ones((len(sequence),3))

    # 反归一化
    predicted_seq = np.concatenate((temp_array_1, pre_temperature, temp_array_2), axis=1)
    predicted_seq = scaler.inverse_transform(predicted_seq)
    pre_temperature = predicted_seq[:,2]
    
    pre_temperature = np.float64(pre_temperature)
    
    return pre_temperature