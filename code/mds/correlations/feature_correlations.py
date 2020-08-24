# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances on the feature scales to the human similarity ratings.

Created on Wed Jan 29 09:51:03 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from itertools import chain, combinations
from code.util import precompute_distances, compute_correlations, distance_functions
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args
   
parser = argparse.ArgumentParser(description='Correlating the scales of two features')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('distance_file', help = 'the pickle file containing the pre-computed distances')
parser.add_argument('output_file', help = 'the csv file to which the output should be saved')
parser.add_argument('-f', '--feature_folder', help = 'folder containing the pickle files for all the features', default = None)
parser.add_argument('-n', '--n_folds', type = int, help = 'number of folds to use for cross-validation when optimizing weights', default = 5)
parser.add_argument('-s', '--seed', type = int, help = 'fixed seed to use for creating the folds', default = None)
add_correlation_metrics_to_parser(parser)
args = parser.parse_args()

correlation_metrics = get_correlation_metrics_from_args(args)

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

items = input_data['items']
target_dissimilarities = input_data['dissimilarities']

if args.feature_folder is not None:
    # load feature data
    feature_data = {}
    for file_name in os.listdir(args.feature_folder):
        if file_name.endswith('.pickle'):
            feature_name = file_name.split('.')[0]
            with open(os.path.join(args.feature_folder, file_name), 'rb') as f_in:
                feature_data[feature_name] = pickle.load(f_in)
    distances = {}
else:
    # load pre-computed distances
    with open(args.distance_file, 'rb') as f_in:
        distances = pickle.load(f_in)

# see https://docs.python.org/3/library/itertools.html
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) 

with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("n_dims,type,dims,scoring,weights,{0}\n".format(','.join(correlation_metrics)))

    # look at the power set of all spaces
    spaces = powerset(sorted(feature_data.keys()))
    for space in spaces:
        
        print('-'.join(space))
        number_of_dimensions = len(space)        
        if number_of_dimensions == 0:
            # ignore empty set
            continue
        
        largest_set_of_scale_types = []
        for feature_name in space:
            if len(feature_data[feature_name].keys()) > len(largest_set_of_scale_types):
                largest_set_of_scale_types = sorted(feature_data[feature_name].keys())
        
        for scale_type in largest_set_of_scale_types:       
        
            if args.feature_folder is not None:
                # populate the vectors
                vectors = []
                
                for item in items:
                    
                    item_vec = []
                    for feature_name in space:
                        if scale_type in feature_data[feature_name]:
                            item_vec.append(feature_data[feature_name]['aggregated'][scale_type][item])
                        else:
                            # features extracted from categories: only have one constant scale type
                            item_vec.append(feature_data[feature_name]['aggregated']['metadata'][item])
                    item_vec = np.array(item_vec)
                    vectors.append(item_vec.reshape(1,-1))
               
            # compute correlations
            for distance_function in sorted(distance_functions.keys()):

                if args.feature_folder is not None:
                   # precompute distances and targets based on the feature values
                    precomputed_distances, precomputed_targets = precompute_distances(vectors, target_dissimilarities, distance_function)
                    if space not in distances:
                        distances[space] = {}
                    if scale_type not in distances[space]:
                        distances[space][scale_type] = {}
                    distances[space][scale_type][distance_function] = (precomputed_distances, precomputed_targets)
                else:
                    # simply grab them from the loaded dictionary
                    precomputed_distances, precomputed_targets = distances[space][scale_type][distance_function]
                    
                # raw correlation
                correlation_results = compute_correlations(precomputed_distances, precomputed_targets, distance_function)
                f_out.write("{0},{1},{2},{3},fixed,{4}\n".format(number_of_dimensions, scale_type,
                                                            '-'.join(space), distance_function,
                                                            ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))

                # correlation with optimized weights
                correlation_results = compute_correlations(precomputed_distances, precomputed_targets, distance_function, args.n_folds, args.seed)
                f_out.write("{0},{1},{2},{3},optimized,{4}\n".format(number_of_dimensions, scale_type,
                                                            '-'.join(space), distance_function,
                                                            ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))

                print('\tdone with {0}-{1}; weights: {2}'.format(scale_type, distance_function, correlation_results['weights']))

# output the collected distances if necessary
if args.feature_folder is not None:
    with open(args.distance_file, 'wb') as f_out:
        pickle.dump(distances, f_out)