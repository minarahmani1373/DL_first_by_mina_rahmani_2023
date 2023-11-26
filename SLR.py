# -*- coding: Simple Linear Regression (SLR)*-
"""
Created on Sat Aug 19 19:47:40 2023

@author: Ehsan Khankeshizadeh
"""
#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Spliting the dataset into the training set and testing set
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=1/3, random_state=0)

#Fitting SLR to the training set
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#Predecting the test set results
y_pred=regressor.predict(X_test)

#Visualiziation the training set results
plt.scatter(X_train, y_train, marker='^', c='r')
plt.plot(X_train, regressor.predict(X_train), 'b' )
plt.title('Salary & Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualiziation the tset set results
plt.scatter(X_test, y_test, marker='^', c='g')
plt.plot(X_train, regressor.predict(X_train), 'b' )
plt.title('Salary & Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()












