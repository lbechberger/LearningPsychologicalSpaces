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
from code.util import normalize_direction
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description='Finding interpretable directions')
parser.add_argument('vector_file', help = 'the input pickle file containing the vectors')
parser.add_argument('n_dims', type = int, help = 'the number of dimensions of the underlying space to consider')
parser.add_argument('feature_file', help = 'the pickle file containing the feature information')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
parser.add_argument('-b', '--baseline_file', help = 'path to file with baseline coordinates', default = None)
args = parser.parse_args()

# project a list of vectors onto a given direction
def project_vectors_onto_direction(vectors, direction):
    projected = []
    np_dir = np.array(direction)
    np_dir.reshape(-1)
    for vector in vectors:
        np_vec = np.array(vector)
        np_vec.reshape(-1)
        value = np.inner(np_vec, np_dir)
        projected.append(value)
    return projected

# global dictionary storing all vectors
all_vectors = {}

# read the vectors
with open(args.vector_file, 'rb') as f_in:
    vector_data = pickle.load(f_in)

all_vectors['MDS'] = [vector_data[args.n_dims]]

# read the baseline vectors if applicable
if args.baseline_file is not None:
    with open(args.baseline_file, 'rb') as f_in:
        baseline_data = pickle.load(f_in)
    for key, inner_dict in baseline_data.items():
        all_vectors[key] = inner_dict[args.n_dims]

# load the feature data
with open(args.feature_file, 'rb') as f_in:
    feature_data = pickle.load(f_in)


# iterate over all data sources (i.e., real MDS vectors and baslines)
for data_source in sorted(all_vectors.keys()):

    # iterate over the individual spaces and compute the results
    for space_index, vectors_dict in enumerate(all_vectors[data_source]):

        # transform classification information into sklearn compatible structure
        classification_data = {}
        for feature_type, dataset in feature_data['classification'].items():
            
            positive_examples = [vectors_dict[item] for item in dataset['positive']]
            negative_examples = [vectors_dict[item] for item in dataset['negative']]
            vectors = np.array(positive_examples + negative_examples)
            vectors = vectors.reshape(-1, args.n_dims)
            targets = [1]*len(positive_examples) + [0]*len(negative_examples)
            classification_data[feature_type] = {'vectors': vectors, 'targets': targets}
            
        # transform regression data into sklearn compatible structure
        regression_data = {}
        for feature_type, dataset in feature_data['aggregated'].items():
        
            vectors = []
            targets = []
            for item, target in dataset.items():
                vectors.append(vectors_dict[item])
                targets.append(target)
            vectors = np.array(vectors)
            vectors = vectors.reshape(-1, args.n_dims)
            regression_data[feature_type] = {'vectors': vectors, 'targets': targets}
            
        # go thorugh each of the data sets
        for feature_type in sorted(classification_data.keys()):
            
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
            kappa_results = {}
            spearman_results = {}
            
            # classification
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
                    kappa = cohen_kappa_score(predictions, classification_data[feature_type]['targets'])
                    max_kappa = max(kappa, max_kappa)
                kappa_results[direction_name] = max_kappa
            
            # correlation
            for direction_name, vectors in projected_vectors_regression.items():
                spearman_results[direction_name], _ = spearmanr(vectors, regression_data[feature_type]['targets'])


            # finally: output results
            # write headline if necessary
            if not os.path.exists(args.output_file):
            
                # construct headline
                headline = 'dims,data_source,space_idx,feature_type,model,kappa,spearman,{0}\n'.format(','.join(['d{0}'.format(i) for i in range(20)]))
                
                with open(args.output_file, 'w') as f_out:
                    fcntl.flock(f_out, fcntl.LOCK_EX)
                    f_out.write(headline)
                    fcntl.flock(f_out, fcntl.LOCK_UN)
            
            # write content
            with open(args.output_file, 'a') as f_out:
                fcntl.flock(f_out, fcntl.LOCK_EX)
                for direction_name in sorted(candidate_directions.keys()):
                    kappa = kappa_results[direction_name]
                    spearman = spearman_results[direction_name]
                    direction = candidate_directions[direction_name]
                    line_items = [args.n_dims, data_source, space_index, feature_type, direction_name, kappa, spearman] + [i for i in direction]
                    f_out.write(",".join(map(lambda x: str(x), line_items)))
                    f_out.write("\n")
                fcntl.flock(f_out, fcntl.LOCK_UN)