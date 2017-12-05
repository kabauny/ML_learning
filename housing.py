#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:38:23 2017

@author: zhongningchen
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit as sss

DOWNLOAD_ROOT = 'http://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'

def fetch_housing_data(housing_url= HOUSING_URL, housing_path = HOUSING_PATH):
    #downloading files and creating the directory. 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()



def load_housing_data(housing_path = HOUSING_PATH):
    #loading the file
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
#creating test set
#########################################################################################

#this function is essentially found in sklearn.model_selsection.StratifiedShuffleSplit
def split_train_test(data, test_ratio):
    #returns training and test data sets given data and test ratio. 
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#train_set2, test_set2 = split_train_test(data = housing, test_ratio = .2)

#the next two functions are probaly OK to keep 
def test_set_check(identifier, test_ratio, hash):
    #in: integer, ratio of test/train distribution, hash function
    #out: returns boolean, if hash of input integer's last element is less than test_ratio
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio

def split_train_set_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

'''
#creating and renaming a new column in the dataframe
housing_with_id = housing.reset_index()
#column with arbitary function for values i.e. longiude*1000 + latutude
housing_with_id['index'] = housing['longitude']*1000 + housing['latitude']
housing_with_id = housing_with_id.rename(columns = {'index': 'id'})
train_set, test_set = split_train_set_by_id(data= housing_with_id, test_ratio = .2, id_column = 'id')

'''

#########################################################################################

housing["income_cat"] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace = True)
split = sss(n_splits = 1, test_size = .2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    







