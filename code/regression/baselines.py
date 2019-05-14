# -*- coding: utf-8 -*-
"""
Implements the baselines against which we compare the linear regression.

Created on Wed Feb  7 10:56:46 2018

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

parser = argparse.ArgumentParser(description='Running the simple baselines')
parser.add_argument('targets_file', help = 'pickle file containing the regression targets')
parser.add_argument('output_folder', help = 'folder where output is stored in form of csv files')
parser.add_argument('-n', '--n_samples', type = int, help = 'number of samples for random baselines', default = 10)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

if args.seed is not None:
   np.random.seed(args.seed)


with open(args.targets_file, 'rb') as f:
    dictionary = pickle.load(f)

space_names = sorted(dictionary.keys())

for space_name in space_names:
    # baselines are identical for correct and shuffled targets --> only compute once
    targets = dictionary[space_name]['targets']
    
    image_names = sorted(targets.keys())
    
    mse_list = {'mean': [], 'zero': [], 'distribution': [], 'draw': []}
    rmse_list = {'mean': [], 'zero': [], 'distribution': [], 'draw': []}
    r2_list = {'mean': [], 'zero': [], 'distribution': [], 'draw': []}

    def evaluate(target, prediction, name):
        mse_list[name].append(mean_squared_error(target, prediction))
        rmse_list[name].append(np.sqrt(mean_squared_error(target, prediction)))
        r2_list[name].append(r2_score(target, prediction))
    
    with open(os.path.join(args.output_folder, '{0}.csv'.format(space_name)), 'w') as f:
      
        f.write('baseline,mse,rmse,r2\n')
        # leave-one-out evaluation    
        for test_image in image_names:
            
            test_vector = targets[test_image]
            train_vectors = [targets[img_name] for img_name in image_names if img_name != test_image]
            
            # prediction of 'mean' baseline
            pred_mean = np.mean(train_vectors, axis=0)   
            evaluate(test_vector, pred_mean, 'mean')        
            
            # prediction of 'zero' baseline
            pred_zero = [0]*len(test_vector)        
            evaluate(test_vector, pred_zero, 'zero')        
            
            # prediction of 'distribution' baseline
            covariance_matrix = np.cov(train_vectors, rowvar = False)
            pred_distr = np.random.multivariate_normal(pred_mean, covariance_matrix, size = args.n_samples)
            for pred in pred_distr:
                evaluate(test_vector, pred, 'distribution')        
            
            # prediction of 'draw' baseline
            indices_draw = np.random.choice(range(len(train_vectors)), size = args.n_samples)
            for idx in indices_draw:
                evaluate(test_vector, train_vectors[idx], 'draw')        
    
        for baseline in ['mean', 'zero', 'distribution', 'draw']:
            mse = np.mean(mse_list[baseline])
            rmse = np.mean(rmse_list[baseline])
            r2 = np.mean(r2_list[baseline])
            f.write('{0},{1},{2},{3}\n'.format(baseline, mse, rmse, r2))