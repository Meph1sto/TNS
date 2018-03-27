#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:50:47 2018

@author: robert_heeley

use XGBoost to rank feature importance

"""

import pandas as pd
from os import getcwd
from os.path import join
import xgboost as xgb

# Set up file paths
curr_path = getcwd()
data_path = join(curr_path, 'data')
out_path = join(curr_path, 'outputs')

# Read in files 
#train_orig = pd.read_csv(join(data_path,'train.csv'))
file_name = 'features_extracted.csv'
output_file = 'feature_importance.csv'

# set up features
features = pd.read_csv(join(out_path, file_name))
train = features[features['fault_severity'] >= 0].copy()
print (train.shape)
print (train.head())
feature_names = list(train.columns)
feature_names.remove('id')
feature_names.remove('fault_severity')
feature_names.remove('location_id')
feature_names.remove('log_order')
fs = ['f%i' % i for i in range(len(feature_names))]

# xgb params
params = {'min_child_weight': 3, 
              'eta': 0.05, 
              'colsample_bytree': 0.4, 
              'max_depth': 10, 
              'subsample': 0.9, 
              'lambda': 0.5,
              'nthread': 8, 
              'objective': 'multi:softprob', 
              'silent': 0, 
              'num_class': 3}

dtrain = xgb.DMatrix(train[feature_names].values, label=train['fault_severity'].values, missing=-999.0, feature_names=fs)
model = xgb.train(params, dtrain, 1800)
feature_imp = model.get_fscore()
f1 = pd.DataFrame({'f': feature_imp.keys(), 'imp': feature_imp.values()})
f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
feature_importance = pd.merge(f1, f2, how='right', on='f')
feature_importance.to_csv(join(out_path, output_file))