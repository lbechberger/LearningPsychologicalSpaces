# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances from the MDS space to the human similarity ratings.

Created on Thu Dec  6 11:43:04 2018

@author: lbechberger
"""

import pickle, argparse, os
from code.util import compute_correlations, distance_functions, load_mds_vectors
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args

parser = argparse.ArgumentParser(description='correlation of MDS distances to human similarity ratings')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('mds_folder', help = 'the folder containing the MDS vectors')
parser.add_argument('-o', '--output_file', help = 'the csv file to which the output should be saved', default='mds.csv')
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

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("n_dims,scoring,weights,{0}\n".format(','.join(correlation_metrics)))

    for number_of_dimensions in range(args.n_min, args.n_max + 1):
        
        # load the vectors into a list 
        vectors = load_mds_vectors(os.path.join(args.mds_folder, "{0}D-vectors.csv".format(number_of_dimensions)), item_ids)
        
        for distance_function in sorted(distance_functions.keys()):

            # raw correlation
            correlation_results = compute_correlations(vectors, target_dissimilarities, distance_function)
            f_out.write("{0},{1},fixed,{2}\n".format(number_of_dimensions, distance_function,
                                                                    ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))

            # correlation with optimized weights
            correlation_results = compute_correlations(vectors, target_dissimilarities, distance_function, args.n_folds, args.seed)
            f_out.write("{0},{1},optimized,{2}\n".format(number_of_dimensions, distance_function,
                                                                    ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))
            print('done with {0}-{1}; weights: {2}'.format(number_of_dimensions, distance_function, correlation_results['weights']))

