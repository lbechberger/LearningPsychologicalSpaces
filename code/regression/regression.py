# -*- coding: utf-8 -*-
"""
Runs the specified regressions and stores the results in the given file.

Created on Tue May 14 10:12:54 2019

@author: lbechberger
"""

import argparse, pickle, os, fcntl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso

parser = argparse.ArgumentParser(description='Regression from feature space to MDS space')
parser.add_argument('targets_file', help = 'pickle file containing the regression targets')
parser.add_argument('space', help = 'name of the target space to use')
parser.add_argument('features_file', help = 'pickle file containing the feature vectors')
parser.add_argument('output_file', help = 'csv file for outputting the results')
parser.add_argument('-z', '--zero', action = 'store_true', help = 'compute zero baseline')
parser.add_argument('-m', '--mean', action = 'store_true', help = 'compute mean baseline')
parser.add_argument('-n', '--normal', action = 'store_true', help = 'compute normal distribution baseline')
parser.add_argument('-d', '--draw', action = 'store_true', help = 'compute random draw baseline')
parser.add_argument('-r', '--repetitions', type = int, help = 'number of repetitions', default = 1)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-l', '--linear', action = 'store_true', help = 'compute linear regression')
parser.add_argument('-a', '--alpha', type = float, help = 'compute lasso regression using the given value of alpha', default = None)
args = parser.parse_args()

# avoid computing multiple regressiontypes in a single run
if sum([args.zero, args.mean, args.normal, args.draw, args.linear, args.alpha is not None]) != 1:
    raise Exception('Must use exactly one regression or baseline type!')

# set seed if needed
if args.seed is not None:
   np.random.seed(args.seed)

# load regression targets
with open(args.targets_file, 'rb') as f:
    targets_dict = pickle.load(f)[args.space]

# load features
with open(args.features_file, 'rb') as f:
    features_dict = pickle.load(f)

if set(targets_dict['correct'].keys()) != set(features_dict.keys()):
    raise Exception('Targets and features do not match!')

if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("regressor,targets,data_set,mse,rmse,r2\n")
        fcntl.flock(f, fcntl.LOCK_UN)

# helper function for computing the three evaluation metrics
def evaluate(ground_truth, prediction):
    mse = mean_squared_error(ground_truth, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth, prediction)
    return mse, rmse, r2

# run any sklearn-based regression
def sklearn_regression(train_features, train_targets, test_features, test_targets, regressor):
    regressor.fit(train_features, train_targets)
    train_predictions = regressor.predict(train_features)
    test_predictions = regressor.predict(test_features)  
    return train_predictions, test_predictions 

# computing the zero baseline
def zero_baseline(train_features, train_targets, test_features, test_targets):
    regressor =  DummyRegressor(strategy = 'constant', constant = [0]*len(train_targets[0]))
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

# computing the zero baseline
def mean_baseline(train_features, train_targets, test_features, test_targets):
    regressor =  DummyRegressor(strategy = 'mean')
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)
    
# computing the normal distribution baseline
def distribution_baseline(train_features, train_targets, test_features, test_targets):
    
    mean = np.mean(train_targets, axis=0)   
    covariance_matrix = np.cov(train_targets, rowvar = False)
    
    train_predictions = np.random.multivariate_normal(mean, covariance_matrix, size = len(train_targets))
    test_predictions = np.random.multivariate_normal(mean, covariance_matrix, size = len(test_targets))
    
    return train_predictions, test_predictions 

# computing the random draw baseline
def draw_baseline(train_features, train_targets, test_features, test_targets):

    train_indices = np.random.choice(range(len(train_targets)), size = len(train_targets))
    test_indices = np.random.choice(range(len(train_targets)), size = len(test_targets))
    
    train_predictions = [train_targets[i] for i in train_indices]
    test_predictions = [train_targets[i] for i in test_indices]
    
    return train_predictions, test_predictions

# computing a simple linear regression
def linear_regression(train_features, train_targets, test_features, test_targets):
    regressor =  LinearRegression(n_jobs = -1)
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

# computing a lasso regression
def lasso_regression(train_features, train_targets, test_features, test_targets):
    regressor =  Lasso(alpha = args.alpha, random_state = np.random.randint(0,1000), precompute = True)
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

if args.zero:
    prediction_function = zero_baseline
    regressor_name = 'zero baseline'
elif args.mean:
    prediction_function = mean_baseline
    regressor_name = 'mean baseline'
elif args.normal:
    prediction_function = distribution_baseline
    regressor_name = 'normal distribution baseline'
elif args.draw:
    prediction_function = draw_baseline
    regressor_name = 'random draw baseline'
elif args.linear:
    prediction_function = linear_regression
    regressor_name = 'linear regression'
elif args.alpha:
    prediction_function = lasso_regression
    regressor_name = 'lasso regression (alpha = {0})'.format(args.alpha)

for target_type in ['correct', 'shuffled']:
    
    image_names = sorted(targets_dict[target_type].keys())

    # collect all ground truths and predictions here
    train_targets_list = []
    train_predictions_list = []
    test_targets_list = []
    test_predictions_list = []
     
    # do image-based leave-one-out: original images determine fold structure
    for test_image in image_names:
    
        # preparing test set for this fold
        test_features = features_dict[test_image]
        target = targets_dict[target_type][test_image]
        test_targets = [target]*len(test_features)
        
        # preparing training set for this fold
        train_features = []
        train_targets = []
        for img_name in image_names:
            if img_name == test_image:
                continue
            features = features_dict[img_name]
            target = targets_dict[target_type][img_name]
            train_features += features
            train_targets += [target]*len(features)
        
        for i in range(args.repetitions):
            train_predictions, test_predictions = prediction_function(train_features, train_targets, test_features, test_targets)

            train_targets_list += train_targets
            train_predictions_list += list(train_predictions)
            test_targets_list += test_targets
            test_predictions_list += list(test_predictions)

    # compute metrics in the end to make sure that R² is reasonable
    train_mse, train_rmse, train_r2 = evaluate(train_targets_list, train_predictions_list)
    test_mse, test_rmse, test_r2 = evaluate(test_targets_list, test_predictions_list)
       
    with open(args.output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('{0},{1},{2},{3},{4},{5}\n'.format(regressor_name, target_type, 'training', train_mse, train_rmse, train_r2))
        f.write('{0},{1},{2},{3},{4},{5}\n'.format(regressor_name, target_type, 'test', test_mse, test_rmse, test_r2))
        fcntl.flock(f, fcntl.LOCK_UN)