# -*- coding: utf-8 -*-
"""
Simple linear regression on the feature vectors.

Created on Tue Jan 30 11:07:57 2018

@author: lbechberger
"""

import sys
import pickle
from math import sqrt
from random import shuffle
from configparser import RawConfigParser
import fcntl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np

options = {}
options['features_file'] = 'features/images'
options['targets_file'] = 'features/targets-4d'
options['features_size'] = 2048
options['space_size'] = 4

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")

if config.has_section(config_name):
    options['features_file'] = config.get(config_name, 'features_file')
    options['targets_file'] = config.get(config_name, 'targets_file')
    options['features_size'] = config.getint(config_name, 'features_size')
    options['space_size'] = config.getint(config_name, 'space_size')

try:
    input_data = pickle.load(open(options['features_file'], 'rb'))
    targets_data = pickle.load(open(options['targets_file'], 'rb'))
except Exception as e:
    print("Cannot read input data. Aborting.")
    print(e)
    sys.exit(0)

real_targets = targets_data['targets']
shuffled_targets = targets_data['shuffled']

squared_train_errors = {'real': [], 'shuffled': []}
squared_test_errors = {'real': [], 'shuffled': []}

for test_image in input_data.keys():
    
    print("Test image {0}".format(test_image))
    train_image_names = [img_name for img_name in input_data.keys() if img_name != test_image]
    
    features_train = []
    real_labels_train = []
    shuffled_labels_train = []
    for img_name in train_image_names:
        features_train += input_data[img_name]
        real_labels_train += [real_targets[img_name]]*len(input_data[img_name])
        shuffled_labels_train += [shuffled_targets[img_name]]*len(input_data[img_name])
    
    zipped = list(zip(features_train, real_labels_train, shuffled_labels_train))
    shuffle(zipped)
    features_train = np.array(list(map(lambda x: x[0], zipped)))
    labels_train = {}
    labels_train['real'] = list(map(lambda x: x[1], zipped))
    labels_train['shuffled'] = list(map(lambda x: x[2], zipped))
    
    features_test = np.array(input_data[test_image])
    labels_test = {}
    labels_test['real'] = [real_targets[img_name]]*len(input_data[test_image])
    labels_test['shuffled'] = [shuffled_targets[img_name]]*len(input_data[test_image])
    
    # reduce number of features
    features_train = features_train[:,:options['features_size']]    
    features_test = features_test[:,:options['features_size']]    
    
    def train_regression(label_type):
        regr = linear_model.LinearRegression(n_jobs=-1)
        regr.fit(features_train, labels_train[label_type])
        train_predictions = regr.predict(features_train)
        test_predictions = regr.predict(features_test)
        squared_train_errors[label_type].append(mean_squared_error(train_predictions, labels_train[label_type]))
        squared_test_errors[label_type].append(mean_squared_error(test_predictions, labels_test[label_type]))
    
    # first train real
    train_regression('real')
    # now train shuffled
    train_regression('shuffled')
    

def rmse(mse_list):
    average_mse = (1.0 * sum(mse_list)) / len(mse_list)
    return sqrt(average_mse)

for label_type in ['real', 'shuffled']:
    combination_label = "{0}-{1}".format(config_name, label_type)
    train_rmse = rmse(squared_train_errors[label_type])
    test_rmse = rmse(squared_test_errors[label_type])
    print("Train RMSE for {0}: {1}".format(combination_label, train_rmse))
    print("Test RMSE for {0}: {1}".format(combination_label, test_rmse))
    
    with open("regression/{0}".format(combination_label), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("{0},{1}\n".format(train_rmse, test_rmse))
        fcntl.flock(f, fcntl.LOCK_UN)
