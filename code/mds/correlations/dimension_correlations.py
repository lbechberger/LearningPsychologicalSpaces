# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances on the interpretable dimensions to the human similarity ratings.

Created on Wed Jan 29 09:51:03 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from code.util import compute_correlations, distance_functions
from itertools import chain, combinations
   
parser = argparse.ArgumentParser(description='Correlating the scales of two dimensions')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('dimension_folder', help = 'folder containing the pickle files for all the dimensions')
parser.add_argument('-o', '--output_file', help = 'the csv file to which the output should be saved', default='dims.csv')
args = parser.parse_args()

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

# load dimension data
dimension_data = {}
for file_name in os.listdir(args.dimension_folder):
    if file_name.endswith('.pickle'):
        dimension_name = file_name.split('.')[0]
        with open(os.path.join(args.dimension_folder, file_name), 'rb') as f_in:
            dimension_data[dimension_name] = pickle.load(f_in)

# see https://docs.python.org/3/library/itertools.html
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) 

with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("n_dims,type,dims,scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")

    # look at the power set of all spaces
    spaces = powerset(sorted(dimension_data.keys()))
    for space in spaces:
        
        number_of_dimensions = len(space)        
        if number_of_dimensions == 0:
            # ignore empty set
            continue
        
        largest_set_of_scale_types = []
        for dim_name in space:
            if len(dimension_data[dim_name].keys()) > len(largest_set_of_scale_types):
                largest_set_of_scale_types = sorted(dimension_data[dim_name].keys())
        
        for scale_type in largest_set_of_scale_types:       
        
            # populate the vectors
            vectors = []
            
            for item_id in item_ids:
                
                item_vec = []
                for dim_name in space:
                    if scale_type in dimension_data[dim_name]:
                        item_vec.append(dimension_data[dim_name][scale_type][item_id])
                    else:
                        # dimensions extracted from categories: only have one constant scale type
                        item_vec.append(dimension_data[dim_name]['metadata'][item_id])
                item_vec = np.array(item_vec)
                vectors.append(item_vec.reshape((-1,1)))
               
            # compute correlations
            for distance_name, distance_function in distance_functions.items():

                correlation_metrics = compute_correlations(vectors, target_dissimilarities, distance_function)
                f_out.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(number_of_dimensions, scale_type,
                                                            '-'.join(space), distance_name,
                                                            correlation_metrics['pearson'], 
                                                            correlation_metrics['spearman'], 
                                                            correlation_metrics['kendall'], 
                                                            correlation_metrics['r2_linear'], 
                                                            correlation_metrics['r2_isotonic']))