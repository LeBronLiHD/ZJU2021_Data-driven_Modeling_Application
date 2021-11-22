# -*- coding: utf-8 -*-

"""
analysis the relation ship between single feature and the label
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from scipy import signal
import load_data
import parameters
import preprocess
import math
import calculation


def resample_data(x_train, y_train):
    index = [i for i in range(len(x_train))]
    index = resample(index, replace=False,
                     n_samples=round(len(x_train) * parameters.G_SampleRatio),
                     random_state=len(x_train[0]))
    x_train = x_train[index]
    y_train = y_train[index]
    print("resample done!")
    return x_train, y_train


def single_analysis(x_train, y_train, show_image=True):
    if len(x_train) != 0:
        num_feature = len(x_train[0])
    else:
        num_feature = -1
        print("Error! x_train is empty!")
    print("number of features ->", num_feature)
    correlation = [[], []]
    x_train, y_train = resample_data(x_train, y_train)
    x_train_t = x_train.T

    for i in range(num_feature):
        print("X(feature) index ->", i, "/ 6")
        correlation[0].append(i)
        correlation[1].append(calculation.get_correlation(x_train_t[i], y_train))
        if show_image:
            plt.figure(figsize=(18, 8),
                       dpi=100)
            plt.subplot(1, 6, 1)
            plt.boxplot(x_train_t[i],
                        notch=False,
                        vert=True,
                        whis=1.5,
                        widths=0.75,
                        patch_artist=True,
                        labels=["X feature " + str(i)])
            plt.subplot(1, 6, 2)
            plt.boxplot(y_train,
                        notch=False,
                        vert=True,
                        whis=1.5,
                        widths=0.75,
                        patch_artist=True,
                        labels=["Y"])
            plt.subplot(1, 6, 3)
            plt.violinplot(x_train_t[i],
                           vert=True,
                           widths=0.75,
                           showmeans=True,
                           showextrema=True,
                           showmedians=True)
            plt.subplot(1, 6, 4)
            plt.violinplot(y_train,
                           vert=True,
                           widths=0.75,
                           showmeans=True,
                           showextrema=True,
                           showmedians=True)
            plt.subplot(1, 6, 5)
            seaborn.distplot(x_train_t[i],
                             hist=True,
                             kde=False,
                             bins=int(180 / 10),
                             color='blue',
                             hist_kws={'edgecolor': 'black'})
            plt.subplot(1, 6, 6)
            seaborn.distplot(y_train,
                             hist=True,
                             kde=False,
                             bins=int(180 / 10),
                             color='blue',
                             hist_kws={'edgecolor': 'black'})
            plt.show()


if __name__ == '__main__':
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
    x_train, y_train, x_test = preprocess.fill_nan_with_zero(x_train), \
                               preprocess.fill_nan_with_zero(y_train), \
                               preprocess.fill_nan_with_zero(x_test)
    single_analysis(x_train, y_train, show_image=True)
