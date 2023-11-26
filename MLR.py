# -*- coding: Multiple Linear Regression (MLR) *-
"""
Created on Sat Aug 19 20:58:11 2023

@author: Ehsan Khankeshizadeh
"""
#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder', OneHotEncoder(), [3] )], remainder='passthrough')
X=np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the dummy varible trap
X=X[:,1:]

#Spliting the dataset into the training set and testing set
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

#Fitting MLR to the training set
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#Predecting the test set results
y_pred=regressor.predict(X_test)





