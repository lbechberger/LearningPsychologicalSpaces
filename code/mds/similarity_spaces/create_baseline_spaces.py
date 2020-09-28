# -*- coding: utf-8 -*-
"""
Create random baseline spaces for later comparison with MDS results.

Created on Mon Sep  7 14:20:40 2020

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from itertools import zip_longest
from code.util import normalize_vectors

parser = argparse.ArgumentParser(description='Creating random baseline spaces')
parser.add_argument('input_pickle_file', help = 'path to the input pickle file containing the preprocessed individual ratings')
parser.add_argument('output_pickle_file', help = 'path to the output pickle file')
parser.add_argument('n_spaces', type = int, help = 'number of example spaces to generate')
parser.add_argument('max_dims', type = int, help = 'maximum number of dimensions to consider')
parser.add_argument('-n', '--normal', action = 'store_true', help = 'use normal distribution')
parser.add_argument('-u', '--uniform', action = 'store_true', help = 'use uniform distribution')
parser.add_argument('-m', '--shuffled', nargs = '*', help = 'list of pickle files for obtaining shuffled coordinates of MDS spaces')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generator', default = None)
args = parser.parse_args()

b_shuffle = False if args.shuffled is None else len(args.shuffled) > 0

if sum([args.normal, args.uniform, b_shuffle]) == 0:
    raise Exception("At least one distribution type must be selected!")

# grab list of items
with open(args.input_pickle_file, 'rb') as f_in:
    input_data = pickle.load(f_in)
    items = sorted(input_data['items'].keys())

output = {}

# first: normally distributed points
if args.normal:

    # set seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)

    space_map = {}
    for n_dim in range(1, args.max_dims + 1):
        space_list = []
        for n_space in range(args.n_spaces):
            space = {}
            coordinates = np.random.normal(size=(len(items), n_dim, 1))
            coordinates = normalize_vectors(coordinates)

            for idx, item in enumerate(items):
                space[item] = coordinates[idx]
            space_list.append(space)
            
        space_map[n_dim] = space_list

    output['normal'] = space_map

# second: uniformly distributed points
if args.uniform:

    # set seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)

    space_map = {}
    for n_dim in range(1, args.max_dims + 1):
        space_list = []
        for n_space in range(args.n_spaces):
            space = {}
            coordinates = np.random.rand(len(items), n_dim, 1)
            coordinates = normalize_vectors(coordinates)

            for idx, item in enumerate(items):
                space[item] = coordinates[idx]
            space_list.append(space)
            
        space_map[n_dim] = space_list

    output['uniform'] = space_map


# thirdly: shuffled versions of actual MDS vectors
if b_shuffle:
    
    def grouper(n, iterable, fillvalue=None):
        "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)    

    for name, file_name in grouper(2, args.shuffled):

        # set seed for reproducibility
        if args.seed is not None:
            np.random.seed(args.seed)
        
        with open(file_name, 'rb') as f_in:
            mds_input = pickle.load(f_in)
            
        space_map = {}
        for n_dim in range(1, args.max_dims + 1):
            space_list = []
            original_coordinates = np.array([mds_input[n_dim][item] for item in items])
            for n_space in range(args.n_spaces):
                space = {}
                coordinates = np.copy(original_coordinates)
                np.random.shuffle(coordinates)
                
                for idx, item in enumerate(items):
                    space[item] = coordinates[idx]
                space_list.append(space)
                
            space_map[n_dim] = space_list
    
        output['shuffled_{0}'.format(name)] = space_map
        
# dump the result in a pickle file
with open(args.output_pickle_file, 'wb') as f_out:
    pickle.dump(output, f_out)