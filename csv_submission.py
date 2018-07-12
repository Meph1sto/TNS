#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:17:50 2018


Run XGBoost model and 10 fold cross val and create Kaggle file for submission

"""

import pandas as pd
import numpy as np
from os import getcwd
from os.path import join
import xgboost as xgb

# Set up file paths
curr_path = getcwd()
data_path = join(curr_path, 'data')
out_path = join(curr_path, 'outputs')

#%%
# xgb params

num_fold = 10
num_jobs = 12

params = {'min_child_weight': 3,
              'eta': 0.04,
              'colsample_bytree': 0.5,
              'max_depth': 8,
              'subsample': 1.0,
              'lambda': 0.5,
              'gamma' : 0.001, 
              'nthread': num_jobs, 
              'objective': 'multi:softprob', 
              'silent': 1, 
              'num_class': 3, 
              'eval_metric' : 'mlogloss', 
              'booster' : 'gbtree', 
              'nrounds': 1800}

#%%
# Set up which features to use for model

# calc logloss for submission
def calc_result(result):
    for row in result.index:
        result_df = []
        print (result)
        iteration = int(row)+1
        test_logloss = np.float(result.iloc[row,0])
        test_logloss_std = np.float(result.iloc[row,1])
        result_df.append([iteration, test_logloss, test_logloss_std])
    result_df = pd.DataFrame(result_df, columns=['i', 'mlogloss', 'std'])
    return result_df

# cross val
cv_result = []

# load files
feature_file_name = 'features_extracted.csv'
features = pd.read_csv(join(out_path, feature_file_name))
feature_importance = pd.read_csv(join(out_path, 'feature_importance.csv'))
feature_importance = feature_importance.fillna(0)

# drop low gain features
drop_cols = list(feature_importance[feature_importance['imp'] < 1]['feature_name'])
print ('drop low gain features', len(drop_cols))

train = features[features['fault_severity'] >= 0].copy() # either 0, 1 or 2
test = features[features['fault_severity'] <0].copy() # was -1 to indicate empty
print (train.shape, test.shape)

# set up features to use
feature_names = list(train.columns)
feature_names.remove('id')
feature_names.remove('fault_severity')
feature_names.remove('location_id')
feature_names.remove('log_order')
feature_names = list(set(feature_names) - set(drop_cols))
print ('features to use', len(feature_names))

#%% 
# Create submission.csv file for Kaggle submission

# change file name below as required
submission_file_name = 'submission.csv'

# run 10 fold cross validation
xtrain = xgb.DMatrix(train[feature_names].values, label=train['fault_severity'].values, missing=-999.0)
result = xgb.cv(params, xtrain, 500, nfold=10, metrics={'mlogloss'}, seed=0)
result_df = calc_result(result)
result_df = result_df.sort_values(by='mlogloss')
best_res = result_df.iloc[0]
num_round = int(best_res['i'])
cv_result.append([feature_file_name] + params.values() + list(best_res))
cv_result_df = pd.DataFrame(cv_result, columns=['feature_file_name'] + params.keys() + list(best_res.index))
print cv_result_df

train['cv'] = np.random.randint(0, 10, len(train))
train_predictions = []
for cv in range(10):
    train_train = train[train['cv'] != cv].copy()
    train_test = train[train['cv'] == cv].copy()
    print (train_train.shape, train_test.shape)
    xtrain_train = xgb.DMatrix(train_train[feature_names].values, label=train_train['fault_severity'].values, missing=-999.0)
    xtrain_test = xgb.DMatrix(train_test[feature_names].values, missing=-999.0)
    xtest = xgb.DMatrix(test[feature_names].values, missing=-999.0)
    model = xgb.train(params, xtrain_train, num_round)
    train_prediction = pd.DataFrame(model.predict(xtrain_test), columns=['predict_0', 'predict_1', 'predict_2'])
    train_prediction['id'] = train_test['id'].values
    train_prediction = train_prediction[['id', 'predict_0', 'predict_1', 'predict_2']]
    train_predictions.append(train_prediction)
    test_prediction = pd.DataFrame(model.predict(xtest), columns=['predict_0', 'predict_1', 'predict_2'])
    if cv == 0:
        test_predictions = test_prediction / 10
    else:
        test_predictions = test_predictions + test_prediction / 10

# send final predictions to .csv file
test_predictions['id'] = test['id'].values
test_predictions = test_predictions[['id', 'predict_0', 'predict_1', 'predict_2']]
test_predictions.to_csv(join(out_path, submission_file_name), index=False)
print (test_predictions.shape)
