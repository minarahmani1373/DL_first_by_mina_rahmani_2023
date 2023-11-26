# -*- coding: Polynomial Linear Regression (PLR) -*-
"""
Created on Sat Aug 19 22:36:37 2023

@author: Ehsan Khankeshizadeh
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#PLR
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Nemudar
plt.scatter(X,y,marker='^', c='b')
plt.plot(X, lin_reg.predict(X), c='r')
plt.title('Graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y,marker='^', c='b')
plt.plot(X, lin_reg2.predict(X_poly), c='g')
plt.title('Graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Pishbini yek morede khas
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))













