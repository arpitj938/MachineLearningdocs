#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:27:56 2017

@author: abhik

Basics functions for handeling pandas

(DataPreprocessing)
"""

#we usually input csv in dataframes using pandas library 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
