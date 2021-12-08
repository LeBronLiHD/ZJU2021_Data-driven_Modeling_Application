# -*- coding: utf-8 -*-

import parameters
import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import preprocess
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
import random
import time
import model_validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import sys

sys.dont_write_bytecode = True

matplotlib.use('TkAgg')


def Train_CNN_Model(x_train, y_train, width, height):
    print("width  ->", width)
    print("height ->", height)
    print("x_train.type  ->", type(x_train))
    print("y_train.type  ->", type(y_train))
    print("x_train.shape ->", x_train.shape)
    print("y_train.shape ->", y_train.shape)

    # building a linear stack of layers with the sequential model
    dropout_param = 0.2
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(2, 2),
                     padding='same',
                     data_format='channels_last',
                     input_shape=(width, height, 1),
                     activation='relu'))
    model.add(Dropout(dropout_param))
    model.add(Conv2D(filters=32,
                     kernel_size=(2, 2),
                     padding='same',
                     data_format='channels_last',
                     input_shape=(width, height, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(dropout_param))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_param))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_param))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_param))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_param))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    early_stopping = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.000001, patience=25, mode='min')
    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=parameters.G_EpochNum,
                        # callbacks=[early_stopping],
                        batch_size=64,
                        shuffle=True)

    print(model.summary())

    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model RMSE')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model_name = "model_forest_cnn_" + str(parameters.G_EpochNum) + ".h5"
    model_path = os.path.join(parameters.G_ModelSave_Sub, model_name)
    model.save(model_path)
    print("model saved at", model_path)

    return model


if __name__ == '__main__':
    path = parameters.G_DataPath_Sub
    x_train, y_train, x_test = load_data.load_train_data(path, forest=False)
    x_train = preprocess.transfer_x_y(x_train)
    x_test = preprocess.transfer_x_y(x_test)
    x_train, x_test = tf.expand_dims(x_train, 3), tf.expand_dims(x_test, 3)
    # y_train = tf.expand_dims(y_train, -1)
    print("exp_x_train.shape ->", x_train.shape)
    print("exp_x_test.shape  ->", x_test.shape)
    # model = Train_CNN_Model(np.array(x_train), np.array(y_train), x_train.shape[1], x_train.shape[2])
    model = "../model/model_forest_cnn_" + str(parameters.G_EpochNum) + ".h5"
    model_validation.model_validation_cnn(model, x_train, y_train, x_test, model_or_path=False)
