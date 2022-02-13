# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:26:54 2020

@author: Yichen Jiang
"""

"""
--- --- this is file is for running time series model on donation data of biden ---
--- V3: donation + polling _twitter data ---
--- ADD MORE OPTIONS on DATERANGE---

"""

# In[]
import os
import json
from textblob import TextBlob
import numpy as np
from nltk.stem.snowball import EnglishStemmer
import nltk
import string
from tqdm import tqdm
import math
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import csv
import glob
import preprocessor as p
#from preprocessor.api import clean, tokenize, parse
from textblob import TextBlob
import statsmodels.api as sm

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
#import regressor
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# In[]
createVar = locals()


# In[]
"""
--- import processed fec data ---

"""
path_fec = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Fundraising data\2019-2020\amount and number'
filenames_fec = os.listdir(path_fec)


# In[]

dict_fec = {}
for filename in filenames_fec:
    if filename.endswith('.csv'):
        dict_fec[filename.strip('.csv')] = pd.read_csv(os.path.join(path_fec,filename),index_col = 0)

"""
--- replace nan with 0 ---

"""
for key in dict_fec.keys():
    dict_fec[key]=dict_fec[key].where(dict_fec[key].notnull(), 0)

"""
--- remove kamala harris ---

"""
for key in dict_fec.keys():
    del dict_fec[key]['harris']

# In[]
"""

--- time series model ---
--- using dict_fec data --- (only donation data)

"""
candidate = 'biden'

df_fec = pd.concat([dict_fec['donation_amount'][str(candidate)], dict_fec['donor_num'][str(candidate)],\
                    dict_fec['newdonation_amount'][str(candidate)],dict_fec['newdonor_num'][str(candidate)]], axis=1)

df_fec.columns = list(dict_fec.keys())


""" --- remove the dates before 2018 --- """

for index in df_fec.index:
    if '2019' not in index:
        df_fec = df_fec.drop([index])

""" --- remove dates prior than start date --- """

for index in df_fec.index:
    if index < '2019-04-18':
        df_fec = df_fec.drop([index])
        
# In[]
"""

--- uses average metrcis of past 3/7/10/14 days ---

"""
dict_average = {}

list_days = [3,7,10,14]

# dates
for date in tqdm(df_fec.index):
    # variables
    for column in df_fec.columns:
        # calculate date range
        for daterange in list_days:
            # if beginning of the date range is within the dataset
            initial_date = (datetime.strptime(date, '%Y-%m-%d')+timedelta(days=-daterange)).strftime('%Y-%m-%d') 
            if initial_date in df_fec.index:
                # create key and dict in dict_average
                if 'ave_'+str(column)+'_'+str(daterange) not in dict_average.keys():
                    dict_average['ave_'+str(column)+'_'+str(daterange)] = {}
                # calculate sum
                sum = 0
                for day in range(0,daterange):
                    past_date = (datetime.strptime(initial_date, '%Y-%m-%d')+timedelta(days=day)).strftime('%Y-%m-%d') 
                    sum += df_fec[column][past_date]
                # calculate average
                dict_average['ave_'+str(column)+'_'+str(daterange)][date] = sum/daterange


# In[]

"""

--- average metrics of future 3/7/10/14 days ---

"""

dict_dependent = {}
list_days = [2,3,4,5,7,10,14]

# dates
for date in tqdm(df_fec.index):
    # variables
    for column in df_fec.columns:
        # calculate date range
        for daterange in list_days:
            # if the end of the date range is within the dataset
            end_date = (datetime.strptime(date, '%Y-%m-%d')+timedelta(days=daterange-1)).strftime('%Y-%m-%d') 
            if end_date in df_fec.index:
                # create key and dict in dict_dependent
                if 'ave_'+str(column)+'_'+str(daterange) not in dict_dependent.keys():
                    dict_dependent['ave_'+str(column)+'_'+str(daterange)] = {}
                # calculate sum
                sum = 0
                for day in range(0,daterange):
                    future_date = (datetime.strptime(date, '%Y-%m-%d')+timedelta(days=day)).strftime('%Y-%m-%d') 
                    sum += df_fec[column][future_date]
                # calculate average
                dict_dependent['ave_'+str(column)+'_'+str(daterange)][date] = sum/daterange


# In[]
# dict -> df
df_average = pd.DataFrame.from_dict(dict_average).dropna()
""" --- dependent var ranges from 2019-04-18 - 2019-12-18 --- """
df_dependent = pd.DataFrame.from_dict(dict_dependent).dropna()[:'2019-12-18'] # [:'2019-12-18'] (remove the last few points)


df_data = df_average[:df_dependent.index[-1]].dropna()

# In[]

"""
--- import daily polling data ---

"""
df_poll = pd.read_csv(os.path.join(r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science','Poll','biden_538.csv'),index_col = 0)


""" poll data start from 03/15/2019 """
df_data = pd.concat([df_average,df_poll['Poll']],axis=1,sort=True)[:df_dependent.index[-1]].dropna()

# In[]

"""
--- import twitter data ---

"""

df_twitter = pd.read_csv(os.path.join(r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Political Science\Debate Data\Biden','df_twitter_biden.csv'),index_col = 0)

""" --- replace nan in df with 0 --- """
""" --- effective twitter data starts from 2019-06-03 --- """

df_twitter = df_twitter.loc['2019-06-03':].fillna(0)


""" complete twitter data start from 06/03/2019 """

df_data = pd.concat([df_data,df_twitter],axis=1,sort=True).dropna()

# In[]

""" --- OR!! dataset of all independent variables starting on 01/15/2019 --- """

#daterange = 7
#initial_date = (datetime.strptime('2019-01-01', '%Y-%m-%d')+timedelta(days=daterange)).strftime('%Y-%m-%d') 
"""
list_columns = []
for daterange in list_days:
    for column in df_fec.columns:
        list_columns.append('ave_'+str(column)+'_'+str(daterange))
"""        
daterange = 10

# choose the dependent variable
var = 'newdonation_amount' # 'donation_amount' 'donor_num' 'newdonation_amount' 'newdonor_num'
dependent_variable = 'ave_'+str(var)+'_'+str(daterange)

start_date = df_data.index[0]

list_test = list(df_data['2019-10-01':'2019-10-31'].index)
list_train = list(df_data.index.difference(list_test))



X_train = df_data.loc[list_train]#[list_columns]
y_train = df_dependent.loc[list(X_train.index)][str(dependent_variable)]


X_test = df_data.loc[list_test]#[list_columns]
y_test = df_dependent.loc[list(X_test.index)][str(dependent_variable)].dropna()

# In[]


"""
--- comparison between regressors 

"""       


#tscv = TimeSeriesSplit(n_splits=8)

models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
models.append(('KNN', KNeighborsRegressor())) 
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
# Evaluate each model in turn
results = []
names = []

for name, model in models:
    
    # n splits for data in training set
    n_splits = math.floor(len(X_train)/30)-1
    tscv = TimeSeriesSplit(n_splits=n_splits)
    """--- choose r2 for the scoring ---"""
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error',error_score='raise')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
fig = plt.figure(dpi = 25, figsize = (50, 35))
plt.boxplot(results, labels=names)
plt.yticks(fontsize = 50)
plt.xticks(fontsize = 50)
plt.title('Algorithm Comparison',  fontdict={'size':'50'})
plt.show()

# In[]
""" --- apply GridSearchCV: selecting the best parameters for the model --- """

""" --- random forest --- """
model = RandomForestRegressor()
param_search = { 
    'n_estimators': [20, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [i for i in range(5,15)]
}
tscv = TimeSeriesSplit(n_splits=n_splits)

# create rmse calculation for gridsearchcv
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

rmse_score = metrics.make_scorer(rmse, greater_is_better = False)


gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

# In[]
""" --- apply GridSearchCV: selecting the best parameters for the model --- """

""" --- SVR --- """
model = SVR()
param_search = {'kernel':('linear', 'rbf'), 
                'C':[1, 2, 4],
                'gamma':['auto']}

tscv = TimeSeriesSplit(n_splits=n_splits)

# create rmse calculation for gridsearchcv
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

rmse_score = metrics.make_scorer(rmse, greater_is_better = False)


gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

# In[]
""" --- apply GridSearchCV: selecting the best parameters for the model --- """

""" --- NN: MLPRegressor --- """
model = MLPRegressor()

pipeline_steps = [('scaler', StandardScaler()), ('MLPR', MLPRegressor(solver='lbfgs', early_stopping=True, validation_fraction=0.1, max_iter=10000))]

param_search = {'MLPR__hidden_layer_sizes':[(50,), (100,), (50,50)], 'MLPR__alpha':[0.001, 10, 1000]}

MLPR_pipeline = Pipeline(pipeline_steps)

tscv = TimeSeriesSplit(n_splits=n_splits)

# create rmse calculation for gridsearchcv
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

rmse_score = metrics.make_scorer(rmse, greater_is_better = False)

gsearch = GridSearchCV(estimator=MLPR_pipeline, cv=tscv, param_grid=param_search, scoring = rmse_score)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_


# In[]
""" --- apply GridSearchCV: selecting the best parameters for the model --- """

""" --- knn --- """
model = KNeighborsRegressor()
param_search = { 
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'metric' : ['euclidean','manhattan']
}
tscv = TimeSeriesSplit(n_splits=n_splits)

# create rmse calculation for gridsearchcv
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

rmse_score = metrics.make_scorer(rmse, greater_is_better = False)


gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

# In[]
""" --- apply GridSearchCV: Selecting the best parameters for the model --- """

""" --- lasso regression --- """
model = Lasso()
# create rmse calculation for gridsearchcv
def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

tscv = TimeSeriesSplit(n_splits=n_splits)

rmse_score = metrics.make_scorer(rmse, greater_is_better = False)

dict_alpha = {'alpha':[0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20]}

gsearch = GridSearchCV(Lasso(), dict_alpha, scoring=rmse_score, cv=tscv )
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_
best_alpha = gsearch.fit(X_train,y_train).best_params_

# In[]
""" --- linear regression --- """
best_model = LinearRegression().fit(X_train, y_train)

# In[]
""" --- define function for checking and comparing results --- """

def regression_results(y_true, y_pred):
# Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


""" calculate prediction results """

y_true = y_test.values
y_pred = best_model.predict(X_test)
regression_results(y_true, y_pred)


# In[]
# plot test
plt.style.use('ggplot')
fig = plt.figure(dpi = 80, figsize = (20, 10))
index = np.arange(len(y_test))
plt.plot(index, y_test,color='k',label='test')
plt.plot(index, y_pred,color='r',label='pred')
plt.xticks(index, y_test.index.values,fontsize = 16,rotation=45)
plt.legend()
plt.show()

