# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances from the MDS space to the human similarity ratings.

Created on Thu Dec  6 11:43:04 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from code.util import precompute_distances, compute_correlations, distance_functions
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args

parser = argparse.ArgumentParser(description='correlation of MDS distances to human similarity ratings')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('distance_file', help = 'the pickle file containing the pre-computed distances')
parser.add_argument('output_file', help = 'the csv file to which the output should be saved')
parser.add_argument('-v', '--vector_file', help = 'the pickle file containing the MDS vectors', default = None)
parser.add_argument('-b', '--baseline_file', help = 'the pickle file containing the random baseline configurations', default = None)
parser.add_argument('--n_min', type = int, default = 1, help = 'the smallest space to investigate')
parser.add_argument('--n_max', type = int, default = 20, help = 'the largest space to investigate')
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

use_precomputed_distances = args.vector_file is None and args.baseline_file is None

all_vectors = {}

if use_precomputed_distances:
    # load pre-computed distances
    with open(args.distance_file, 'rb') as f_in:
        distances = pickle.load(f_in)
    data_sources = sorted(distances.keys())
else:
    distances = {}
    
    if args.vector_file is not None:
        # load vectors
        with open(args.vector_file, 'rb') as f_in:
            mds_data = pickle.load(f_in)
        all_vectors['MDS'] = {}
        for number_of_dimensions in range(args.n_min, args.n_max + 1):
            all_vectors['MDS'][number_of_dimensions] = [mds_data[number_of_dimensions]]
        
    if args.baseline_file is not None:
        # load random configurations
        with open(args.baseline_file, 'rb') as f_in:
            baseline_data = pickle.load(f_in)
        for baseline, inner_dict in baseline_data.items():
            all_vectors[baseline] = inner_dict
    data_sources = sorted(all_vectors.keys())


with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("n_dims,data_source,scoring,weights,{0}\n".format(','.join(correlation_metrics)))

    for data_source in data_sources:    
        for number_of_dimensions in range(args.n_min, args.n_max + 1):

            list_of_raw_results = {}
            list_of_weighted_results = {}            
            
            if use_precomputed_distances:
                number_of_examples = len(distances[data_source][number_of_dimensions])
            else:
                number_of_examples = len(all_vectors[data_source][number_of_dimensions])
            
            for i in range(number_of_examples):
                
                if not use_precomputed_distances:
                    vector_list = []
                    for item in items:
                        vector_list.append(all_vectors[data_source][number_of_dimensions][i][item])
            
                for distance_function in sorted(distance_functions.keys()):

                    if use_precomputed_distances:
                        # use pre-computed distances
                        precomputed_distances = distances[data_source][number_of_dimensions][distance_function][i]
                    else:
                        # pre-compute distances and store them
                        precomputed_distances = precompute_distances(vector_list, distance_function)
                        if data_source not in distances:
                            distances[data_source] = {}
                        if number_of_dimensions not in distances[data_source]:
                            distances[data_source][number_of_dimensions] = {}
                        if distance_function not in distances[data_source][number_of_dimensions]:
                            distances[data_source][number_of_dimensions][distance_function] = {}
                        distances[data_source][number_of_dimensions][distance_function][i] = precomputed_distances
                    
    
                    # raw correlation
                    raw_correlation_results = compute_correlations(precomputed_distances, target_dissimilarities, distance_function)
                    if distance_function not in list_of_raw_results:
                        list_of_raw_results[distance_function] = []
                    list_of_raw_results[distance_function].append(raw_correlation_results)
    
                    # correlation with optimized weights
                    weighted_correlation_results = compute_correlations(precomputed_distances, target_dissimilarities, distance_function, args.n_folds, args.seed)
                    if distance_function not in list_of_weighted_results:
                        list_of_weighted_results[distance_function] = []
                    list_of_weighted_results[distance_function].append(weighted_correlation_results)
                
            for distance_function in sorted(distance_functions.keys()):
                
                aggregated_raw_results = {}
                for metric in correlation_metrics:
                    aggregated_raw_results[metric] = np.mean(list(map(lambda x: x[metric], list_of_raw_results[distance_function])))
                f_out.write("{0},{1},{2},fixed,{3}\n".format(number_of_dimensions, data_source, distance_function,
                                                                        ','.join(map(lambda x: str(aggregated_raw_results[x]), correlation_metrics))))
    
                aggregated_weighted_results = {}
                for metric in correlation_metrics:
                    aggregated_weighted_results[metric] = np.mean(list(map(lambda x: x[metric], list_of_weighted_results[distance_function])))
                aggregated_weights = np.mean(list(map(lambda x: x['weights'], list_of_weighted_results[distance_function])), axis = 0)
                f_out.write("{0},{1},{2},optimized,{3}\n".format(number_of_dimensions, data_source, distance_function,
                                                                        ','.join(map(lambda x: str(aggregated_weighted_results[x]), correlation_metrics))))
                print('done with {0}-{1}-{2}; weights: {3}'.format(number_of_dimensions, data_source, distance_function, aggregated_weights))

# output the collected distances if necessary
if args.vector_file is not None:
    with open(args.distance_file, 'wb') as f_out:
        pickle.dump(distances, f_out)