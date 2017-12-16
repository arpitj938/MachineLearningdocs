#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:57:56 2017

@author: abhik

Template to get the improve the input 

(DataPreprocessing)
"""


###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################

#example of taking csv input
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

###############################################################################

