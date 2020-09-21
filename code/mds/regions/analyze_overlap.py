# -*- coding: utf-8 -*-
"""
Analyze the overlap between conceptual regions.

Created on Wed Nov 14 10:57:29 2018

@author: lbechberger
"""

import pickle, argparse, os, fcntl
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser(description='Analyze overlap of conceptual regions')
parser.add_argument('input_file', help = 'the input pickle file containing the vectors and the category structure')
parser.add_argument('n_dims', type = int, help = 'dimensionality of space to investigate')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
parser.add_argument('-b', '--baseline_file', help = 'path to file with baseline coordinates', default = None)
args = parser.parse_args()

# function that checks whether this point is a convex combination of the hull points
# based on https://stackoverflow.com/a/43564754
def in_hull(hull_points, candidate_point):
    n_points = len(hull_points)
    c = np.zeros(n_points)
    A = np.r_[hull_points.T,np.ones((1,n_points))]
    b = np.r_[candidate_point, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

# count the number of violation for the given hull and candidate points
def count_violations(hull_points, candidate_points):
    counter = 0
    for candidate_point in candidate_points:
        if in_hull(hull_points, candidate_point):
            # a point from a different category lies within the convex hull, violating the convexity criterion
            counter += 1
    return counter

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
all_violations = {}
for hull_category_type in ['VC', 'VV', 'all']:
    all_violations[hull_category_type] = {}
    for intruder_category_type in ['VC', 'VV', 'all']:
        all_violations[hull_category_type][intruder_category_type] = {}
        for data_source in all_vectors.keys():
            all_violations[hull_category_type][intruder_category_type][data_source] = 0


# iterate over all category pairs
for hull_category in categories:
    items_in_hull_category = data['categories'][hull_category]['items']
    hull_category_type = data['categories'][hull_category]['visSim']
            
    for intruder_category in categories:
        items_in_intruder_category = data['categories'][intruder_category]['items']
        intruder_category_type = data['categories'][intruder_category]['visSim']
        
        # counts will anyways be zero, so skip
        if intruder_category == hull_category:
            continue
        
        # look at all data sources
        for data_source in sorted(all_vectors.keys()):
            
            num_violations = 0            
            
            for space in all_vectors[data_source]:
        
                # grab the corresponding points 
                hull_points = np.array([space[item] for item in items_in_hull_category]) 
                hull_points = hull_points.reshape((-1, args.n_dims))
                query_points = [space[item].reshape(-1) for item in items_in_intruder_category]    
                
                # count the number of violations: how many items of intruder category are inside hull category?
                num_violations += count_violations(hull_points, query_points)
                
            # need to take the average across all spaces to get expected value for baselines
            num_violations /= len(all_vectors[data_source])

            # add the counts
            for hull_type in [hull_category_type, 'all']:
                for intruder_type in [intruder_category_type, 'all']:
                    all_violations[hull_type][intruder_type][data_source] += num_violations
            
                

# write headline if necessary
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f_out:
        fcntl.flock(f_out, fcntl.LOCK_EX)
        f_out.write("dims,hull_category_type,intruder_category_type,data_source,violations\n")
        fcntl.flock(f_out, fcntl.LOCK_UN)
        

# write content
with open(args.output_file, 'a') as f_out:
    fcntl.flock(f_out, fcntl.LOCK_EX)
    
    for hull_category_type in ['VC', 'VV', 'all']:
        for intruder_category_type in ['VC', 'VV', 'all']:
            for data_source in sorted(all_violations[hull_category_type][intruder_category_type].keys()):
                violations = all_violations[hull_category_type][intruder_category_type][data_source]
                f_out.write("{0},{1},{2},{3},{4}\n".format(args.n_dims, hull_category_type, intruder_category_type, data_source, violations))
    
    fcntl.flock(f_out, fcntl.LOCK_UN)
            