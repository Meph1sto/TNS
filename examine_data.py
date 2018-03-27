#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:02:11 2018

@author: robert_heeley

Get a very quick overview of the data files

"""

import pandas as pd
import numpy as np
from os import getcwd
from os.path import join

# Set up file paths
curr_path = getcwd()
data_path = join(curr_path, 'data')

# Read files and convert to dataframes
train_orig = pd.read_csv(join(data_path,'train.csv'))
test_orig = pd.read_csv(join(data_path, 'test.csv'))
severity_type = pd.read_csv(join(data_path, 'severity_type.csv'))
log_feature = pd.read_csv(join(data_path, 'log_feature.csv'))
event_type = pd.read_csv(join(data_path, 'event_type.csv'))
resource_type = pd.read_csv(join(data_path, 'resource_type.csv'))

file_list = (train_orig, test_orig, severity_type, log_feature, event_type, resource_type)
feat_list = ('id', 'location', 'fault_severity', 'severity_type', 'log_feature', 'volume', 'event_type', 'resource_type')

# Print df overview
def print_details(file_list):
    for data in file_list:
        print data.shape
        print data.head(), data.info(), data.describe()
        print '\n'

# Check files for empty or NaN values
def check_for_null(file_list):
    for data in file_list:
        print np.where(pd.isnull(data))
        print np.where(data.applymap(lambda x: x == ''))
        print '\n'

# Check files for unique values
def get_unique_values(file_list):
    for data in file_list:
        if 'location' in data: 
            print data.groupby('location')['id'].nunique()

# Output to console
print_details(file_list)
check_for_null(file_list)
get_unique_values(file_list)       

'''
Saved outputs of df heads

(7381, 3)
      id      location  fault_severity
0  14121  location 118               1
1   9320   location 91               0
2  14394  location 152               1
3   8218  location 931               1
4  14804  location 120               0


(11171, 2)
      id      location
0  11066  location 481
1  18000  location 962
2  16964  location 491
3   4795  location 532
4   3392  location 600


(18552, 2)
     id    severity_type
0  6597  severity_type 2
1  8011  severity_type 2
2  2597  severity_type 2
3  5022  severity_type 1
4  6852  severity_type 1


(58671, 3)
     id  log_feature  volume
0  6597   feature 68       6
1  8011   feature 68       7
2  2597   feature 68       1
3  5022  feature 172       2
4  5022   feature 56       1


(31170, 2)
     id     event_type
0  6597  event_type 11
1  8011  event_type 15
2  2597  event_type 15
3  5022  event_type 15
4  5022  event_type 11


(21076, 2)
     id    resource_type
0  6597  resource_type 8
1  8011  resource_type 8
2  2597  resource_type 8
3  5022  resource_type 8
4  6852  resource_type 8
'''