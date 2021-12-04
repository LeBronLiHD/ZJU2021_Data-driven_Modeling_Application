# -*- coding: utf-8 -*-

"""
evaluate the model
"""

import numpy
import parameters
from keras.models import load_model

# @TODO: def model_evaluate(model, data_x, data_y)

def model_validation(x_train, y_train, ratio=parameters.G_SampleRatio):

