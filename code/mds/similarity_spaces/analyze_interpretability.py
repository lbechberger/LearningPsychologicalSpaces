# -*- coding: utf-8 -*-
"""
Checks the interpretability of the space by trying to train linear SVMs.

Created on Fri Nov 16 12:47:16 2018

@author: lbechberger
"""

import argparse, os, pickle, fcntl
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import cohen_kappa_score, r2_score
import numpy as np
from code.util import load_mds_vectors

parser = argparse.ArgumentParser(description='Finding interpretable directions')
parser.add_argument('vector_file', help = 'the input csv file containing the vectors')
parser.add_argument('n_dims', type = int, help = 'the number of dimensions in the underlying space')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
parser.add_argument('-c', '--classification_file', help = 'the pickle file containing the classification information')
parser.add_argument('-r', '--regression_file', help = 'the pickle file containing the regression information')
parser.add_argument('-b', '--baseline', action = "store_true", help = 'whether or not to compute the random baselines')
parser.add_argument('-n', '--repetitions', type = int, help = 'number of repetitions in sampling the baselines', default = 20)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation when computing baselines and for SVM', default = None)
args = parser.parse_args()

# make sure we deal with either regresison or classification
if sum([args.classification_file is not None, args.regression_file is not None]) != 1:
    raise Exception('Must use exactly one type of ML task!')

# load classification data and transform it into sklearn compatible structure
def prepare_classification_data(vectors_dict):
    with open(args.classification_file, 'rb') as f_in:
        raw_data = pickle.load(f_in)
    
    classification_data = {}
    for dataset_name, dataset in raw_data.items():
        
        positive_examples = [vectors_dict[item] for item in dataset['positive']]
        negative_examples = [vectors_dict[item] for item in dataset['negative']]
        vectors = positive_examples + negative_examples
        targets = [1]*len(positive_examples) + [0]*len(negative_examples)
        
        classification_data[dataset_name] = {'vectors': vectors, 'targets': targets}
    
    return classification_data

# load regression data and transform it into sklearn compatible structure
def prepare_regression_data(vectors_dict):
    with open(args.regression_file, 'rb') as f_in:
        raw_data = pickle.load(f_in)
    
    regression_data = {}
    for dataset_name, dataset in raw_data.items():

        vectors = []
        targets = []        
        for item, target in dataset.items():
            vectors.append(vectors_dict[item])
            targets.append(target)

        regression_data[dataset_name] = {'vectors': vectors, 'targets': targets}        
        
    return regression_data

# SVC for classification
def get_classification_model():
    return LinearSVC(dual = False, random_state = args.seed)

# SVR for regression
def get_regression_model():
    return LinearSVR(dual = False, loss='squared_epsilon_insensitive', random_state = args.seed)

if args.classification_file is not None:
    # classification
    load_data = prepare_classification_data
    get_model = get_classification_model
    metric = cohen_kappa_score
    metric_name = "kappa"
else:
    # regression
    load_data = prepare_regression_data
    get_model = get_regression_model
    metric = r2_score
    metric_name = "r2"
   
# read the vectors
vectors = load_mds_vectors(args.vector_file)

if args.seed is not None:
    np.random.seed(args.seed)

# load the data from the pickle file
datasets = load_data(vectors)

# go thorugh each of the data sets
for dataset_name, dataset in datasets.items():
    
    # train and evaluate model
    model = get_model()
    model.fit(dataset['vectors'], dataset['targets'])
    direction = np.reshape(model.coef_, (-1))
    result = metric(model.predict(dataset['vectors']), dataset['targets'])
    
    result_uniform = 0
    result_normal = 0
    result_shuffled = 0    

    # compute baselines if necessary
    if args.baseline:
    
        for i in range(args.repetitions):
            # unformly distributed points
            uniform_points = np.random.rand(len(dataset['vectors']), args.n_dims)
            uniform_model = get_model()
            uniform_model.fit(uniform_points, dataset['targets'])
            result_uniform += metric(uniform_model.predict(uniform_points), dataset['targets'])       
            
            # normally distributed points
            normal_points = np.random.normal(size=(len(dataset['vectors']), args.n_dims))
            normal_model = get_model()
            normal_model.fit(normal_points, dataset['targets'])
            result_normal += metric(normal_model.predict(normal_points), dataset['targets'])       
            
            # shuffled points
            shuffled_points = np.array(list(dataset['vectors']))
            np.random.shuffle(shuffled_points)
            shuffled_model = get_model()
            shuffled_model.fit(shuffled_points, dataset['targets'])
            result_shuffled += metric(shuffled_model.predict(shuffled_points), dataset['targets'])       
        
        # average across repetitions to get expected value
        result_uniform /= args.repetitions
        result_normal /= args.repetitions
        result_shuffled /= args.repetitions

      
    # write headline if necessary
    if not os.path.exists(args.output_file):

        # construct headline
        headline = 'dims,dataset,{0}_mds,{0}_u,{0}_n,{0}_s,{1}\n'.format(metric_name, ','.join(['d{0}'.format(i) for i in range(20)]))
        
        with open(args.output_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(headline)
            fcntl.flock(f, fcntl.LOCK_UN)
    
    # write content
    with open(args.output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)                
        line_items = [args.n_dims, dataset_name, result, result_uniform, result_normal, result_shuffled] + [i for i in direction]
        f.write(",".join(map(lambda x: str(x), line_items)))
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)