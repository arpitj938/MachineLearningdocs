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


#total number of rows and columns
dataset.shape   

#if you want to see last few rows then use tail command (default last 5 rows will print)
dataset.tail()

#slicing
dataset[2:5]

dataset.columns

#priting particular column data 
dataset.column
dataset['column'] #dataset.column (both are same)
dataset[['column1', 'column2']] #getting two or more column at once 



dataset['column'].max()
dataset['column'].min() 
dataset['column'].describe()

# select rows which has maximum value
dataset[dataset.column == dataset.column.max()] 


