#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:34:56 2017

@author: abhik

Basics functions for Numpy

(DataPreprocessing)
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.array([0, 1, 2, 3])

a.ndim

a.shape

# arange is an array-valued version of the built-in Python range function
a = np.arange(10)

#using linspace
a = np.linspace(0, 1, 6) #start, end, number of points

a = np.ones((3, 3))
#The default data type is float for zeros and ones function

"""
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
"""

b = np.zeros((3, 3))


c = np.eye(3)  #Return a 2-D array with ones on the diagonal and zeros elsewhere.

a = np.diag([1, 2, 3, 4]) #construct a diagonal array.

"""
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])
"""


a = np.random.rand(4) 
# array([ 0.57944059,  0.4826708 ,  0.66348841,  0.37141847])



x = np.array([1, 2, 3, 4])
np.sum(x)

x.sum(axis=0)
x.max()
x.min()
#index of minimum element
x.argmin()
#index of maximum element
x.argmax()

a = np.array([[1, 2, 3], [4, 5, 6]])
#Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
a.ravel() 

#Transpose Used quite extensively 
a.T  

b = b.reshape((2, 3))

a = np.array([[5, 4, 6], [2, 3, 2]])
b = np.sort(a, axis=1)