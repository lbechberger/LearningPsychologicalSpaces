# -*- coding: utf-8 -*-
"""
Checks the interpretability of the space by trying to train linear SVMs.

Created on Fri Nov 16 12:47:16 2018

@author: lbechberger
"""

import argparse, os, pickle, fcntl
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
import numpy as np
from code.util import load_mds_vectors, normalize_direction
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description='Finding interpretable directions')
parser.add_argument('vector_file', help = 'the input csv file containing the vectors')
parser.add_argument('n_dims', type = int, help = 'the number of dimensions in the underlying space')
parser.add_argument('classification_file', help = 'the pickle file containing the classification information')
parser.add_argument('regression_file', help = 'the pickle file containing the regression information')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
args = parser.parse_args()

# project a list of vectors onto a given direction
def project_vectors_onto_direction(vectors, direction):
    projected = []
    np_dir = np.array(direction)
    np_dir.reshape(-1, 1)
    for vector in vectors:
        np_vec = np.array(vector)
        np_vec.reshape(-1, 1)
        projected.append(np.inner(vector, direction))#cosine_similarity(vector,direction))
    return projected
        

# read the vectors
vectors_dict = load_mds_vectors(args.vector_file)

# load classification data and transform it into sklearn compatible structure
with open(args.classification_file, 'rb') as f_in:
    raw_data = pickle.load(f_in)

classification_data = {}
for feature_type, dataset in raw_data.items():
    
    positive_examples = [vectors_dict[item] for item in dataset['positive']]
    negative_examples = [vectors_dict[item] for item in dataset['negative']]
    vectors = positive_examples + negative_examples
    targets = [1]*len(positive_examples) + [0]*len(negative_examples)
    
    classification_data[feature_type] = {'vectors': vectors, 'targets': targets}
    
# load regression data and transform it into sklearn compatible structure
with open(args.regression_file, 'rb') as f_in:
    raw_data = pickle.load(f_in)

regression_data = {}
for feature_type, dataset in raw_data.items():

    vectors = []
    targets = []        
    for item, target in dataset.items():
        vectors.append(vectors_dict[item])
        targets.append(target)

    regression_data[feature_type] = {'vectors': vectors, 'targets': targets}        
        
# go thorugh each of the data sets
for feature_type in classification_data.keys():
    
    candidate_directions = {}    
    
    # train linear SVC on classification problem
    svc_model = LinearSVC(dual = False)
    svc_model.fit(classification_data[feature_type]['vectors'], classification_data[feature_type]['targets'])
    svc_direction = np.reshape(svc_model.coef_, (-1))
    candidate_directions['SVC'] = normalize_direction(svc_direction)   
        
    # train linear regression on regression problem    
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(regression_data[feature_type]['vectors'], regression_data[feature_type]['targets'])
    lin_reg_direction = np.reshape(lin_reg_model.coef_, (-1))
    candidate_directions['LinReg'] = normalize_direction(lin_reg_direction)
    
    # now project all data points onto the directions
    projected_vectors_classification = {}
    projected_vectors_regression = {}
    for direction_name, direction in candidate_directions.items():
        projected_vectors_classification[direction_name] = project_vectors_onto_direction(classification_data[feature_type]['vectors'], direction)
        projected_vectors_regression[direction_name] = project_vectors_onto_direction(regression_data[feature_type]['vectors'], direction)
        
    # evaluate by classification (Cohen's kappa) and correlation (Spearman correlation)
    evaluation_kappa = {}
    evaluation_spearman = {}
    
    # classificationo
    for direction_name, vectors in projected_vectors_classification.items():
        # simple threshold classifier
        sorted_vectors = sorted(vectors)
        max_kappa = 0
        
        for i in range(len(sorted_vectors) - 1):
            # test all possible thresholds
            threshold = (sorted_vectors[i] + sorted_vectors[i+1]) / 2
            predictions = []
            for vec in vectors:
                if vec <= threshold:
                    predictions.append(0)
                else:
                    predictions.append(1)
            kappa = cohen_kappa_score(predictions, classification_data[dataset_name]['targets'])
            max_kappa = max(kappa, max_kappa)
        evaluation_kappa[direction_name] = max_kappa
    
    # correlation
    for direction_name, vectors in projected_vectors_regression.items():
        evaluation_spearman[direction_name], _ = spearmanr(vectors, regression_data[dataset_name]['targets'])
        
    # finally: output results
        
    # write headline if necessary
    if not os.path.exists(args.output_file):

        # construct headline
        headline = 'dims,feature_type,model,kappa,spearman,{0}\n'.format(','.join(['d{0}'.format(i) for i in range(20)]))
        
        with open(args.output_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(headline)
            fcntl.flock(f, fcntl.LOCK_UN)
    
    # write content
    with open(args.output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)           
        for direction_name in evaluation_kappa.keys():
            line_items = [args.n_dims, feature_type, direction_name, evaluation_kappa[direction_name], evaluation_spearman[direction_name]] + [i for i in candidate_directions[direction_name]]
            f.write(",".join(map(lambda x: str(x), line_items)))
            f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)