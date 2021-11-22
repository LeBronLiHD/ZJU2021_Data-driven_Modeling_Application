# -*- coding: utf-8 -*-

"""
some APIs for mathematical calculation
"""

import load_data
import parameters
import math
import statistics
import heapq


def get_correlation(x, y):
    cov_up = 0
    stan_1 = 0
    stan_2 = 0
    mean_1 = statistics.mean(x)
    mean_2 = statistics.mean(y)
    for i in range(len(x)):
        cov_up += (x[i] - mean_1) * (y[i] - mean_2)
        stan_1 += (x[i] - mean_1) * (x[i] - mean_1)
        stan_2 += (y[i] - mean_2) * (y[i] - mean_2)
    stan_1, stan_2 = math.sqrt(stan_1), math.sqrt(stan_2)
    return cov_up/(stan_1 * stan_2)

