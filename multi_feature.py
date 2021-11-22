# -*- coding: utf-8 -*-

"""
analysis the relation ship between multi features and the label
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess
import load_data
import parameters
from sklearn.utils import shuffle
import single_feature


def classification(value):
    if value < 0.2:
        return 0
    elif value < 0.5:
        return 1
    else:
        return 2


def transfer_data(x_train, y_train):
    # merge x_train and y_train and return in DataFrame
    data_plot = {
        "1": x_train.T[0],
        "2": x_train.T[1],
        "3": x_train.T[2],
        "4": x_train.T[3],
        "5": x_train.T[4],
        "6": x_train.T[5],
        "7": x_train.T[6],
        "y": y_train
    }
    data_plot = pandas.DataFrame(data_plot)
    print("merge&transfer x_train and y_train done.")
    return data_plot


def heat_map(data):
    print("data.shape ->", data.shape)
    print("data.columns ->", data.columns)
    size = len(data.columns)
    plt.subplots(figsize=(size, size))
    sns.heatmap(data.corr(), annot=True, vmax=1, square=True,
                yticklabels=data.columns.values.tolist(),
                xticklabels=data.columns.values.tolist(), cmap="RdBu")
    plt.title("heatmap")
    plt.show()


def get_top_three(correlation):
    corr_value = []
    for i in range(len(correlation[0])):
        corr_value.append(abs(correlation[1][i]))
    corr_value.sort(reverse=True)
    standard = corr_value[2]
    top_three_corr = [[], []]
    for i in range(len(correlation[0])):
        if abs(correlation[1][i]) >= standard:
            top_three_corr[0].append(correlation[0][i])
            top_three_corr[1].append(correlation[1][i])
    return top_three_corr


def multi_analysis(x_train, y_train, correlation):
    top_three_corr = get_top_three(correlation)
    print("top_three_index ->", top_three_corr[0])
    print("top_three_corr  ->", top_three_corr[1])
    data = transfer_data(x_train, y_train)
    heat_map(data)
    data = pandas.DataFrame(data, columns=top_three_corr[0])
    size = len(data.columns)
    for i in range(len(data)):
        data[data.columns[size - 1]][i] = classification(data[data.columns[size - 1]][i])
    print("multi-feature analysis done.")


if __name__ == '__main__':
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
    x_train, y_train, x_test = preprocess.fill_nan_with_zero(x_train), \
                               preprocess.fill_nan_with_zero(y_train), \
                               preprocess.fill_nan_with_zero(x_test)
    correlation = single_feature.single_analysis(x_train, y_train,
                                                 show_image=False,
                                                 show_corr=False,
                                                 abs_corr=True)
    multi_analysis(x_train, y_train, correlation)
