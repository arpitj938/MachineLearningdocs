#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:43:05 2017

@author: abhik
regression basics

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# first we split the data into test and training set that is standard procedure 
#we can aslo scale the variable depends on the use cases 


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

###############################################################################
"""Simple Linear Refression"""
#basic implemenation only for more details check docs
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

###############################################################################
"""Polynomial Regression"""
#basic implemenation only for more details check docs
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

###############################################################################
"""Random Forest Regressor"""
#basic implemenation only for more details check docs
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
y_pred = regressor.predict(X_test)

###############################################################################
"""Support Vector Regressor"""
#basic implemenation only for more details check docs
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #RBF kenrel other kernel options are available as wee;;
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred) # when we scale we need to inverse to get the result 

###############################################################################
"""Decision Tree Regressor """
#basic implemenation only for more details check docs
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
y_pred = regressor.predict(X_test)
###############################################################################

