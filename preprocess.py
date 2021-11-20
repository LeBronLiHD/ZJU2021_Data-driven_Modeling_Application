# -*- coding: utf-8 -*-

"""
data pre-process
1. data cleaning
    1. missing value
        1. delete the piece of data
        2. interpolation of value
            1. replace
            2. nearest neighbor imputation
            3. regression method
            4. spline interpolation
    2. outliers
        1. simple statistical analysis
            1. observe the maximum and minimum values and determine whether it is reasonable
            2. three delta in normal distribution
            3. box plot analysis
                1. upper quartile and lower quartile
                2. overcome the problem that delta in distribution is under the influence of outliers
    3. duplicated data
        1. analysis first, and remove it if the duplicated data makes no sense
    3. inconsistent data
2. data transformation
    1. square, square root, exponent, logarithm, etc.
    2. normalization
        1. maximum and minimum normalization
        2. zero mean normalization
    3. discretization of continuous data
    4. attribute structure, like BMI
"""
