# -*- coding: utf-8 -*-
"""
Runs the specified regressions and stores the results in the given file.

Created on Tue May 14 10:12:54 2019

@author: lbechberger
"""

import argparse, pickle, os, fcntl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser(description='Regression from feature space to MDS space')
parser.add_argument('targets_file', help = 'pickle file containing the regression targets')
parser.add_argument('space', help = 'name of the target space to use')
parser.add_argument('features_file', help = 'pickle file containing the feature vectors')
parser.add_argument('folds_file', help = 'csv file containing the structre of the folds')
parser.add_argument('output_file', help = 'csv file for outputting the results')
parser.add_argument('--zero', action = 'store_true', help = 'compute zero baseline')
parser.add_argument('--mean', action = 'store_true', help = 'compute mean baseline')
parser.add_argument('--normal', action = 'store_true', help = 'compute normal distribution baseline')
parser.add_argument('--draw', action = 'store_true', help = 'compute random draw baseline')
parser.add_argument('--linear', action = 'store_true', help = 'compute linear regression')
parser.add_argument('--lasso', type = float, help = 'compute lasso regression using the given relative strength of regularization', default = None)
parser.add_argument('--random_forest', action = 'store_true', help = 'compute random forest regression')
parser.add_argument('--shuffled', action = 'store_true', help = 'also train/test on shuffled targets')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

# avoid computing multiple regressiontypes in a single run
if sum([args.zero, args.mean, args.normal, args.draw, args.linear, args.lasso is not None, args.random_forest]) != 1:
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

# make sure targets and features match
if set(targets_dict['correct'].keys()) != set(features_dict.keys()):
    raise Exception('Targets and features do not match!')

# prepare output file if necessary
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("regressor,targets,train_mse,train_med,train_r2,test_mse,test_med,test_r2\n")
        fcntl.flock(f, fcntl.LOCK_UN)

# load the fold structure into a dictionary (fold_id --> list of images in fold)
folds = {}
with open(args.folds_file, 'r') as f:
    for line in f:
        tokens = line.replace('\n', '').split(',')
        if len(tokens) == 2:
            fold = tokens[1]
            image = tokens[0]
            
            if fold not in folds:
                folds[fold] = []
            
            folds[fold].append(image)

# helper function for computing the three evaluation metrics
def evaluate(ground_truth, prediction):
    mse = mean_squared_error(ground_truth, prediction)

    # conversion necessary for mean euclidean distance in 1-dimensional target space
    gt = np.array(ground_truth)
    p = np.array(prediction)
    if len(p.shape) == 1:
        p = p.reshape(-1,1)
    if len(gt.shape) == 1:
        gt = gt.reshape(-1,1)
    mean_euclidean_distance = np.mean(paired_distances(gt, p, metric = 'euclidean'))

    r2 = r2_score(ground_truth, prediction)
    return mse, mean_euclidean_distance, r2

# run any sklearn-based regression
def sklearn_regression(train_features, train_targets, test_features, test_targets, regressor):
    targets = np.array(train_targets)
    if targets.shape[1] == 1:
        targets = targets.ravel()
    regressor.fit(train_features, targets)
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
    regressor =  LinearRegression(normalize = True, n_jobs = -1)
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

# computing a lasso regression
def lasso_regression(train_features, train_targets, test_features, test_targets):
    alpha = args.lasso / len(train_features[0])
    regressor =  Lasso(alpha = alpha, precompute = True, max_iter = 10000, normalize = True)
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

# computing a random forest regression with default hyperparameters
def random_forest_regression(train_features, train_targets, test_features, test_targets):
    regressor =  RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = args.seed)
    return sklearn_regression(train_features, train_targets, test_features, test_targets, regressor)

# collect the features and the targets for the given fold
def prepare_fold(fold_images, features, targets):
    fold_features = []
    fold_targets = []
    for img_name in fold_images:
        img_features = features_dict[img_name]
        img_target = targets[img_name]
        fold_features += img_features
        fold_targets += [img_target]*len(img_features)
    return fold_features, fold_targets


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
elif args.lasso is not None:
    prediction_function = lasso_regression
    regressor_name = 'lasso regression (beta = {0})'.format(args.lasso)
elif args.random_forest:
    prediction_function = random_forest_regression
    regressor_name = 'random forest regression'.format(args.lasso)

target_types = ['correct']
if args.shuffled:
    target_types.append('shuffled')

for target_type in target_types:
    
    image_names = sorted(targets_dict[target_type].keys())
    fold_ids = sorted(folds.keys())

    # collect all ground truths and predictions here
    train_targets_list = []
    train_predictions_list = []
    test_targets_list = []
    test_predictions_list = []
     
    # perform cross validation
    for test_fold in fold_ids:
    
        # which images belong to test and which to train?
        test_images = folds[test_fold]
        train_images = [img_name for img_name in image_names if img_name not in test_images]
        
        # prepare features and targets for train and test
        test_features, test_targets = prepare_fold(test_images, features_dict, targets_dict[target_type])        
        train_features, train_targets = prepare_fold(train_images, features_dict, targets_dict[target_type])        
               
        train_predictions, test_predictions = prediction_function(train_features, train_targets, test_features, test_targets)

        train_targets_list += train_targets
        train_predictions_list += list(train_predictions)
        test_targets_list += test_targets
        test_predictions_list += list(test_predictions)

    # compute metrics in the end to make sure that R² is reasonable
    train_mse, train_med, train_r2 = evaluate(train_targets_list, train_predictions_list)
    test_mse, test_med, test_r2 = evaluate(test_targets_list, test_predictions_list)
       
    with open(args.output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(regressor_name, target_type, train_mse, train_med, train_r2, test_mse, test_med, test_r2))
        fcntl.flock(f, fcntl.LOCK_UN)