# -*- coding: utf-8 -*-
"""
Comparing different interpretable directions based on the cosine similarity.

Created on Sat Feb  1 15:09:35 2020

@author: lbechberger
"""

import argparse, os, csv
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

parser = argparse.ArgumentParser(description='Comparing interpretable directions')
parser.add_argument('input_folder', help = 'the folder containing all input files')
parser.add_argument('n_dims', type = int, help = 'the maximal number of dimensions to consider')
parser.add_argument('output_file', help = 'csv file for output')
args = parser.parse_args()

# one entry for each dimension; then split up into direction_name; then contain list of normal vectors
direction_data = {}
for dims in range(1, args.n_dims + 1):
    direction_data[dims] = {}

# load direction data
for file_name in os.listdir(args.input_folder):
    if file_name.endswith('.csv'):
        direction_name = file_name.split('.')[0]
        for dims in range(1, args.n_dims + 1):
            direction_data[dims][direction_name] = []

        with open(os.path.join(args.input_folder, file_name), 'r') as f_in:            
            reader = csv.DictReader(f_in)
            for row in reader:
                dims = int(row['dims'])
                vector = [float(row['d{0}'.format(d)]) for d in range(dims)]
                direction_data[dims][direction_name].append(vector)

output_dict = {}
categories = []

# look at each similarity space separately
for dims in range(1, args.n_dims + 1):
    
    dims_dict = direction_data[dims]
    output_dict[dims] = {}
    
    # first look at within-similarity for each direction (should be high)
    for direction_name, all_vectors in dims_dict.items():
        sim_matrix = cosine_similarity(all_vectors)
        
        # only take into account entries above the diagonal
        avg_cos_sim = 0
        counter = 0
        for i in range(len(all_vectors)):
            for j in range(i, len(all_vectors)):
                avg_cos_sim += sim_matrix[i][j]
                counter += 1
        avg_cos_sim /= counter
        
        output_dict[dims][direction_name] = avg_cos_sim
        if dims == 1:
            categories.append(direction_name)
    
    # now look at between-similarity for all pairs of directions (should be low)
    for first_direction, second_direction in combinations(sorted(dims_dict.keys()), 2):
        first_vectors = dims_dict[first_direction]
        second_vectors = dims_dict[second_direction]
        all_vectors = first_vectors + second_vectors
        sim_matrix = cosine_similarity(all_vectors)
        
        # only take into account entries above the diagonal, ignoring within-similarities
        avg_cos_sim = 0
        counter = 0
        for i in range(len(first_vectors)):
            for j in range(len(first_vectors), len(first_vectors) + len(second_vectors)):
                avg_cos_sim += sim_matrix[i][j]
                counter += 1
        avg_cos_sim /= counter
        
        category_name = '{0}-{1}'.format(first_direction, second_direction)
        output_dict[dims][category_name] = avg_cos_sim
        if dims == 1:
            categories.append(category_name)

with open(args.output_file, 'w') as f_out:
    # write headline
    headline_items = ['dims'] + categories
    f_out.write(','.join(headline_items))
    f_out.write('\n')
    
    for dims in range(1, args.n_dims + 1):
        line_items = [dims] + [output_dict[dims][category_name] for category_name in categories]
        f_out.write(','.join(map(lambda x: str(x), line_items)))
        f_out.write('\n')