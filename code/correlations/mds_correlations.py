# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances from the MDS space to the human similarity ratings.

Created on Thu Dec  6 11:43:04 2018

@author: lbechberger
"""

import pickle, argparse, os
from code.util import compute_correlations, distance_functions, load_mds_vectors

parser = argparse.ArgumentParser(description='correlation of MDS distances to human similarity ratings')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('mds_folder', help = 'the folder containing the MDS vectors')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('--n_min', type = int, default = 1, help = 'the smallest space to investigate')
parser.add_argument('--n_max', type = int, default = 20, help = 'the largest space to investigate')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots of distances vs. dissimilarities')
args = parser.parse_args()

# set up file name for output file
_, path_and_file = os.path.splitdrive(args.similarity_file)
_, file = os.path.split(path_and_file)
file_without_extension = file.split('.')[0]
output_file_name = os.path.join(args.output_folder, "{0}-MDS.csv".format(file_without_extension))

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

with open(output_file_name, 'w', buffering=1) as f_out:

    f_out.write("n_dims,scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")

    for number_of_dimensions in range(args.n_min, args.n_max + 1):
        
        # load the vectors into a list 
        vectors = load_mds_vectors(os.path.join(args.mds_folder, "{0}D-vectors.csv".format(number_of_dimensions)), item_ids)
        
        for distance_name, distance_function in distance_functions.items():

            correlation_metrics = compute_correlations(vectors, target_dissimilarities, distance_function)
            f_out.write("{0},{1},{2},{3},{4},{5},{6}\n".format(number_of_dimensions, distance_name,
                                                                    correlation_metrics['pearson'], 
                                                                    correlation_metrics['spearman'], 
                                                                    correlation_metrics['kendall'], 
                                                                    correlation_metrics['r2_linear'], 
                                                                    correlation_metrics['r2_isotonic']))