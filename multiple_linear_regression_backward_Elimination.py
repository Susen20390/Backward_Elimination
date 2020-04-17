# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:54:36 2020

@author: SUSA1119
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
print(dataset.head())
print(dataset['State'].unique())


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding Dummy variable Trap

X = X[:,1:]


#Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

# Taking care of missing data
#Class Import
"""from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])"""

# Fitting Multiplle Lineasr Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination


# Optimal matrix of features
import statsmodels.regression.linear_model as smf
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary() 

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary() 

X_opt = X[:, [0, 3, 5]]
regressor_OLS = smf.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary() 


#Backward Elimination with p-values only:

#import statsmodels.formula.api as sm
import statsmodels.api as sm
def backwardElimination(X, sl):
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(X, j, 1)
    regressor_OLS.summary()
    return X
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.api as sm
def backwardElimination(X, SL):
    numVars = len(X[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = X[:, j]
                    X = np.delete(X, j, 1)
                    tmp_regressor = sm.OLS(Y, X).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((X, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return X
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)














