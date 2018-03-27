#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:56:12 2018

@author: robert_heeley

Create and extract features from supplied data files and append them to training and test data

"""

import pandas as pd
import numpy as np
from os import getcwd
from os.path import join

# Set up file paths
curr_path = getcwd()
data_path = join(curr_path, 'data')
out_path = join(curr_path, 'outputs')

# Read files and convert to dataframes
train_orig = pd.read_csv(join(data_path,'train.csv'))
test_orig = pd.read_csv(join(data_path, 'test.csv'))
severity_type = pd.read_csv(join(data_path, 'severity_type.csv'))
log_feature = pd.read_csv(join(data_path, 'log_feature.csv'))
event_type = pd.read_csv(join(data_path, 'event_type.csv'))
resource_type = pd.read_csv(join(data_path, 'resource_type.csv'))

file_list = (train_orig, test_orig, severity_type, log_feature, event_type, resource_type)
feat_list = ('id', 'location', 'fault_severity', 'severity_type', 'log_feature', 'volume', 'event_type', 'resource_type')

#%%

# Set up training and test data to accept new features
#

# Seperate and add location id
train_orig['location_id'] = train_orig.location.apply(lambda x: int(x.split('location ')[1]))

# Add test column with easily removable variable -1
test_orig['fault_severity'] = -1

# Seperate and add location id
test_orig['location_id'] = test_orig.location.apply(lambda x: int(x.split('location ')[1]))
print ('train', train_orig.shape, 'test', test_orig.shape)

# Combine features for simplicity
features = train_orig.append(test_orig)
features = features.drop('location', axis=1)
#print (features.shape)
#print (features.head())

#%%

# Add basic features (from feat_list) on common ids

# Add severity_type 
severity_type['value'] = 1
severity_type_features = severity_type.pivot(index='id', columns='severity_type', values='value')
severity_type_features.columns = [c.replace(' ', '_') for c in severity_type_features.columns]
severity_type_features = severity_type_features.fillna(0)
severity_type_features = severity_type_features[['severity_type_1', 'severity_type_2', 'severity_type_3', 'severity_type_4',
                                                 'severity_type_5']]
features = pd.merge(features, severity_type_features, how='left', left_on='id', right_index=True)
#print (features.shape)
#print (features.head())

# Add event_type and count by id
event_type = pd.read_csv(join(data_path, 'event_type.csv'))
event_count = event_type.groupby('id').count()[['event_type']]
event_count.columns = ['event_type_count']
features = pd.merge(features, event_count, how='inner', left_on='id', right_index=True)
event_type_count = event_type.groupby('event_type').count()[['id']].sort_values(by='id', ascending=False)
frequent_event_types = event_type_count[event_type_count['id'] > 10]
frequent_event_records = event_type[event_type['event_type'].isin(frequent_event_types.index)].copy()
frequent_event_records['value'] = 1
event_features = frequent_event_records.pivot(index='id', columns='event_type', values='value')
event_features.columns = map(lambda x: x.replace(' ', '_'), event_features.columns)
features = pd.merge(features, event_features, how='left', left_on='id', right_index=True)
event_type['event_id'] = event_type.event_type.apply(lambda x: int(x.split('event_type ')[1]))
features = features.fillna(0)
#print (features.shape)
#print (features.head())

# Add resource_type and count by id
resource_type['value'] = 1
resource_type_count = resource_type.groupby('id').count()[['value']]
resource_type_count.columns = ['resource_type_count']
features = pd.merge(features, resource_type_count, how='left', left_on='id', right_index=True)

resource_type_features = resource_type.pivot(index='id', columns='resource_type', values='value')
resource_type_features.columns = [c.replace(' ', '_') for c in resource_type_features.columns]
resource_type_features = resource_type_features[['resource_type_1',  'resource_type_2', 'resource_type_3', 'resource_type_4',
                                                 'resource_type_5','resource_type_6', 'resource_type_7', 'resource_type_8',
                                                 'resource_type_9', 'resource_type_10',]]
features = pd.merge(features, resource_type_features, how='left', left_on='id', right_index=True)
features = features.fillna(0)
#print (features.shape)
#print (features.head())

#%%
# Generate where possible (count, mean, median, sum, min, max, std etc.) on features 

# count log_feature
count_log_feature = log_feature.groupby('id').count()[['log_feature']]
count_log_feature.columns = ['count_log_feature']
features = pd.merge(features, count_log_feature, how='inner', left_on='id', right_index=True)
#print (features.shape)
#print (features.head())

# max log_feature
log_feature['log_feature_id'] = log_feature.log_feature.apply(lambda x: int(x.split('feature ')[1]))
max_log_feature = log_feature.groupby('id').max()[['log_feature_id']]
max_log_feature.columns = ['max_log_feature']
features = pd.merge(features, max_log_feature, how='left', left_on='id', right_index=True)

# median log_feature
median_log_feature = log_feature.groupby('id').median()[['log_feature_id']] 
median_log_feature.columns = ['median_log_feature']
features = pd.merge(features, median_log_feature, how='left', left_on='id', right_index=True)

# mean log_feature
mean_log_feature = log_feature.groupby('id').mean()[['log_feature_id']] 
mean_log_feature.columns = ['mean_log_feature']
features = pd.merge(features, mean_log_feature, how='left', left_on='id', right_index=True)

# min log_feature
min_log_feature = log_feature.groupby('id').min()[['log_feature_id']] 
min_log_feature.columns = ['min_log_feature']
features = pd.merge(features, min_log_feature, how='left', left_on='id', right_index=True)

# count location
location_count = features.groupby('location_id').count()[['id']]
location_count.columns = ['location_count']
features = pd.merge(features, location_count, how='inner', left_on='location_id', right_index=True)

# frequent locations
frequent_locations = location_count[location_count['location_count'] > 10]
frequent_location_records = features[features['location_id'].isin(frequent_locations.index)].copy()
frequent_location_records['value'] = 1
location_features = frequent_location_records.pivot(index='id', columns='location_id', values='value')
location_features.columns = ['location_%i' % c for c in location_features.columns]
print ('location_features', location_features.shape)
features = pd.merge(features, location_features, how='left', left_on='id', right_index=True)

# Add order on log_feature
log_feature_order = log_feature[['id']].drop_duplicates()
log_feature_order['log_order'] = 1. * np.arange(len(log_feature_order)) / len(log_feature_order)
features = pd.merge(features, log_feature_order, how='inner', on='id')
#print (features.shape)
#print (features.head())

# rank loc features by id and log order
features['loc_rank_ascend'] = features.groupby('location_id')[['log_order']].rank()
features['loc_rank_descend'] = features.groupby('location_id')[['log_order']].rank(ascending=False)
features['loc_rank_relative'] = 1. * features['loc_rank_ascend'] / features['location_count']
features['loc_rank_relative'] = np.round(features['loc_rank_relative'], 2)

# fill gaps & print outputs
features = features.fillna(0)
print (features.shape)
print (features.head())
    
#%%
 # Export csv file to outputs folder 
features.to_csv(join(out_path, 'features_extracted.csv'))
print ('final features', features.shape)
#%%