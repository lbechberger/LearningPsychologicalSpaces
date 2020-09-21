# -*- coding: utf-8 -*-
"""
Analyze the size of the conceptual regions.

Created on Wed Jan 29 12:47:17 2020

@author: lbechberger
"""

import pickle, argparse, os, fcntl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

parser = argparse.ArgumentParser(description='Analyzing the size of conceptual regions')
parser.add_argument('input_file', help = 'the input pickle file containing the vectors and the category structure')
parser.add_argument('n_dims', type = int, help = 'dimensionality of space to investigate')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
parser.add_argument('-b', '--baseline_file', help = 'path to file with baseline coordinates', default = None)
args = parser.parse_args()

# global dictionary storing all vectors
all_vectors = {}

# read the data set
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)
categories = sorted(data['categories'].keys())
all_vectors['MDS'] = [data[args.n_dims]]

# read the baseline vectors if applicable
if args.baseline_file is not None:
    with open(args.baseline_file, 'rb') as f_in:
        baseline_data = pickle.load(f_in)
    for key, inner_dict in baseline_data.items():
        all_vectors[key] = inner_dict[args.n_dims]

# prepare the results dictionary
all_distances = {}
for category_type in ['VC', 'VV', 'all']:
    all_distances[category_type] = {}
    for key in all_vectors.keys():
        all_distances[category_type][key] = []

# iterate over all categories
for category in categories:
    
    # grab category information
    items_in_category = data['categories'][category]['items']
    category_type = data['categories'][category]['visSim']
    
    # run all computations
    for data_source in sorted(all_vectors.keys()):
        
        for space in all_vectors[data_source]:
            
            # collect points belonging to category
            category_points = []    
            for item in items_in_category:  
                vec = np.array(space[item])
                category_points.append(vec.reshape((-1,1)))
                
            # compute average Euclidean distance
            prototype = np.mean(category_points, axis=0)
            distances = [euclidean_distances(point,prototype)[0][0] for point in category_points]
            mean_distance = np.mean(distances)
            
            # store it
            for cat_type in [category_type, 'all']:
                all_distances[cat_type][data_source].append(mean_distance)


# write headline if necessary
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f_out:
        fcntl.flock(f_out, fcntl.LOCK_EX)    
        f_out.write("dims,category_type,data_source,size\n")
        fcntl.flock(f_out, fcntl.LOCK_UN)

# write content
with open(args.output_file, 'a') as f_out:
    fcntl.flock(f_out, fcntl.LOCK_EX)

    for category_type in ['VC', 'VV', 'all']:
        for data_source in sorted(all_distances[category_type].keys()):
            size = np.mean(all_distances[category_type][data_source])
            f_out.write("{0},{1},{2},{3}\n".format(args.n_dims, category_type, data_source, size))
    
    fcntl.flock(f_out, fcntl.LOCK_UN)
        