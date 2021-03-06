# -*- coding: utf-8 -*-

"""
load data for training model
"""

import pandas
import math
import os
import parameters
from pandas import read_csv
from matplotlib import pyplot
import numpy as np


def load_train_data(file_path, forest=False):
    if forest:
        # @TODO load data preprocess in random forest
        print("path ->", file_path, " <forest>")
        train_data, forest_data = [], []
        number_of_train = len(parameters.G_DataTrainFile)
        for i in range(number_of_train):
            train_data.append(os.path.join(file_path, parameters.G_DataTrainFile[i]))
        y_train = np.loadtxt(train_data[1], dtype=np.float32)
        number_of_forest = len(parameters.G_ForestData)
        for i in range(number_of_forest):
            forest_data.append(os.path.join(file_path, parameters.G_ForestData[i]))
        forest_train, forest_test = np.loadtxt(forest_data[0], dtype=np.float32), \
                                    np.loadtxt(forest_data[1], dtype=np.float32)
        print("x_train.shape ->", np.array(forest_train).shape)
        y_train = y_train[parameters.G_DeletionOf_Y:len(y_train)]
        print("y_train.shape ->", y_train.shape)
        print("x_test.shape  ->", np.array(forest_test).shape)
        return np.array(forest_train), y_train, np.array(forest_test)
    else:
        print("path ->", file_path)
        train_data = []
        number_of_train = len(parameters.G_DataTrainFile)
        test_data = []
        number_of_test = len(parameters.G_DataTestFile)
        for i in range(number_of_train):
            train_data.append(os.path.join(file_path, parameters.G_DataTrainFile[i]))
        for i in range(number_of_test):
            test_data.append(os.path.join(file_path, parameters.G_DataTestFile[i]))
        x_train, y_train = np.loadtxt(train_data[0], dtype=np.float32), \
                           np.loadtxt(train_data[1], dtype=np.float32)
        x_test = np.loadtxt(test_data[0], dtype=np.float32)
        print("x_train.shape ->", x_train.shape)
        y_train = y_train[parameters.G_DeletionOf_Y:len(y_train)]
        print("y_train.shape ->", y_train.shape)
        print("x_test.shape  ->", x_test.shape)
        return x_train, y_train, x_test


if __name__ == '__main__':
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_train_data(path, forest=True)
