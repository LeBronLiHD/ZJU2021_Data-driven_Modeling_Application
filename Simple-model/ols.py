# -*- coding: utf-8 -*-

"""
use ols for regression
"""

import load_data
import parameters
import numpy
import os


if __name__ == '__main__':
    path = parameters.G_DataPath
    load_data.load_train_data(path)
