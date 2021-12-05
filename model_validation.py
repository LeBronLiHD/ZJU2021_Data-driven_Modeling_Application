# -*- coding: utf-8 -*-

"""
evaluate the model
"""

import numpy
import numpy as np

import parameters
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K

# @TODO: def model_evaluate(model, data_x, data_y)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def model_validation(model_or_model_path, x_train, y_train, model_or_path=True, ratio=parameters.G_SampleRatio):
    if model_or_path:
        model = model_or_model_path
    else:
        model = load_model(model_or_model_path)
    predict_ori = model.predict(x_train)
    print("prediction of nn done")
    predict = []
    for i in range(len(predict_ori)):
        predict.append(float(predict_ori[i][0]))
    mean_pred = np.mean(predict)
    print("mean od prediction is ->", mean_pred)
    K = [i for i in range(101)]
    R = []
    predict_search = [0 for i in range(len(x_train))]
    for _ in range(len(K)):
        if _ % 10 == 0:
            print("Tuneing ... i =", _)
        for i in range(len(x_train)):
            predict_search[i] = (predict[i] - mean_pred) * K[_] + mean_pred
        predict_search = np.array(predict_search)
        tensor = root_mean_squared_error(y_train, predict_search)
        tensor = float(tensor)
        R.append(tensor)
    print("min RMSE is ->", R[np.argmin(R)])
    best_k = K[np.argmin(R)]
    for i in range(len(x_train)):
        predict[i] = (predict[i] - mean_pred) * best_k + mean_pred
    predict = np.array(predict)
    index = [_ for _ in range(len(predict))]
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(index, y_train, "--b", label="origin")
    plt.plot(index, predict, "-r", label="predict")
    plt.legend(loc="upper left")
    plt.title("origin and prediction in same plot")
    plt.show()
    plt.figure(figsize=(18, 6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(index, y_train, "-b")
    plt.title("origin")
    plt.subplot(1, 2, 2)
    plt.plot(index, predict, "-r")
    plt.title("prediction")
    plt.show()
    print("RMSE  =", root_mean_squared_error(y_train, predict))
    score = model.evaluate(np.array(x_train), np.array(y_train))
    print("score =", score[0])


def model_validation_cnn(model_or_model_path, x_train, y_train, model_or_path=True, ratio=parameters.G_SampleRatio):
    if model_or_path:
        model = model_or_model_path
    else:
        model = load_model(model_or_model_path)
    predict = model.predict(x_train)
    predict = np.array(predict)
    print("prediction of cnn done")
    predict_show, y_train_show, x_train_show = [], [], []
    for i in range(len(x_train)):
        # if i % select_cube == 0:
        predict_show.append(predict[i])
        x_train_show.append(x_train[i])
        y_train_show.append(y_train[i])
    # y_train_ori, predict_ori = y_train, predict
    # x_train, y_train, predict = x_train[select_index], y_train[select_index], predict[select_index]
    index = [_ for _ in range(len(predict_show))]
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(index, y_train_show, "--b", label="origin")
    plt.plot(index, predict_show, "-r", label="predict")
    plt.legend(loc="upper left")
    plt.title("origin and prediction in same plot")
    plt.show()
    plt.figure(figsize=(18, 6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(index, y_train_show, "-b")
    plt.title("origin")
    plt.subplot(1, 2, 2)
    plt.plot(index, predict_show, "-r")
    plt.title("prediction")
    plt.show()
    print("RMSE  =", root_mean_squared_error(y_train, predict))
    score = model.evaluate(np.array(x_train_show), np.array(y_train_show))
    print("score =", score[0])
