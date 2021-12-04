# -*- coding: utf-8 -*-

import parameters
import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import preprocess
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import time
import model_validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve


if __name__ == '__main__':
    path = parameters.G_DataPath
    x_train, y_train, x_test = load_data.load_train_data(path)
