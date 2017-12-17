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

dataset.info()

dataset.columns
dataset.index

#priting particular column data 
dataset.column
dataset['column'] #dataset.column (both are same)
dataset[['column1', 'column2']] #getting two or more column at once 



dataset['column'].max()
dataset['column'].min() 
dataset['column'].describe()

# select rows which has maximum value
dataset[dataset.column == dataset.column.max()] 


##################################################################################################
#Pandas series 

data = dataset['coloumn']
type(data) # pandas.core.series.Series
newNumpyData = data.values #converts Series into Numpy array

users['fees'] = 0 # Broadcasts to entire column





##################################################################################################
#Building DataFrames

#methord 1
data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
'city': ['Austin', 'Dallas', 'Austin', 'Dallas',
'visitors': [139, 237, 326, 456],
'signups': [7, 12, 3, 5]
}

users = pd.DataFrame(data)

#methord 2
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
signups = [7, 12, 3, 5]
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]
list_labels = ['city', 'signups', 'visitors', 'weekday']
list_cols = [cities, signups, visitors, weekdays]
zipped = list(zip(list_labels, list_cols))
data = dict(zipped)
users = pd.DataFrame(data)

##################################################################################################
#writing DataFrames

data.to_csv('output.csv')

##################################################################################################
#Plotting DataFrames
