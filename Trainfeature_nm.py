#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# open dataset from 
with open('newTable_mergeTable.txt', 'r') as f:
	diabetes_train = f.read().splitlines()
f.closed

# define two lists 
diabetes_X_train = []
diabetes_Y_train = []
for line in diabetes_train[:len(diabetes_train)]:
	temp_list = map(int, line.split()[1:-1])
	diabetes_X_train.append(temp_list);
	diabetes_Y_train.append(int(line.split()[-1]))

# normalization
# change diabetes to numpy array 
# and use numpy function to normalize in each column
ft_X_train = np.array(diabetes_X_train)
min_X_train = np.argmin(ft_X_train, axis = 0)
max_X_train = np.argmax(ft_X_train, axis = 0)
range_X_tarin = np.ones(min_X_train.shape)
for i in range(0,14):
	min_X_train[i] = ft_X_train[min_X_train[i]][i]
	max_X_train[i] = ft_X_train[max_X_train[i]][i]
	range_X_tarin[i] = (max_X_train[i] - min_X_train[i])*1.00
	if range_X_tarin[i] == 0.0:
		range_X_tarin[i] = 1.00
# use broadcosting here
ft_X_tarin_nm = (ft_X_train - min_X_train)/range_X_tarin

# output normalized (into [0,1]) train dataset
with open('Trainfeature_nm.txt', 'wb') as f:
	np.savetxt(f,ft_X_tarin_nm)
f.closed

