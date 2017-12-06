#!/usr/bin/env python3
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer as lb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

#######################################################################################

def load_housing_data(housing_path = HOUSING_PATH):
    #loading the file
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)



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

housing = load_housing_data()
#creating a stratified test, train set p52
housing["income_cat"] = np.ceil(housing['median_income']/1.5) 
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace = True)
split = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#del housing['income_cat']


#########################################################################################
#plotting the data for better visualization 

'''
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = .1)
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending = False))
'''

#########################################################################################


#spliting housing into x and y, where label ,y, is median housing value

hX = strat_train_set.drop('median_house_value', axis = 1)
hY = strat_train_set['median_house_value'].copy()


rooms_ix, bedrooms_ix, population_ix, household_ix =  3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:,bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            return np.c_[X,rooms_per_household, population_per_household] 


#this simply returns pd.series of the selected attribute in np.array
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

housing_num = hX.drop('ocean_proximity', axis = 1)
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy = 'median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])

cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label', LabelEncoder()),
            ('1hot', OneHotEncoder()),
            ('')
        
        
        
        ])
    
    
'''
cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label_binarizer', lb())
        ])

full_pipeline = FeatureUnion(transformer_list = [
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline)        
        ])

'''





