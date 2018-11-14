# -*- coding: utf-8 -*-
"""
Analyze a given space for convextiy.

Created on Wed Nov 14 10:57:29 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser(description='MDS for shapes')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
parser.add_argument('data_set_file', help = 'the pickle file containing the data set')
parser.add_argument('n', type = int, help = 'number of dimensions of the MDS space')
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

# read the data set
with open(args.data_set_file, 'rb') as f:
    data_set = pickle.load(f)

# read the vectors
vectors = {}
with open(args.vector_file, 'r') as f:
    for line in f:
        vector = []
        tokens = line.replace('\n', '').split(',')
        # first entry is the item ID
        item = tokens[0]
        # all other entries are the coordinates
        vector += list(map(lambda x: float(x), tokens[1:]))
        vectors[item] = vector

items = list(vectors.keys())
categories = []
for item in items:
    category = data_set['items'][item]['category']
    if category not in categories:
        categories.append(category)

sim_violations = {'Sim':0, 'Dis':0, 'x':0}
artificial_violations = {'art':0, 'nat':0}
sum_violations = 0

uniform_sum_violations = 0
normal_sum_violations = 0
shuffled_sum_violations = 0

for category in categories:
    items_in_category = [item for item in items if item in data_set['categories'][category]['items']]
    items_not_in_category = [item for item in items if item not in items_in_category]
    
    # grab the corresponding points 
    hull_points = np.array([vectors[item] for item in items_in_category]) 
    query_points = [vectors[item] for item in items_not_in_category]    
    
    num_violations = count_violations(hull_points, query_points)
    
    vis_sim = data_set['categories'][category]['visSim']
    artificial = data_set['categories'][category]['artificial']
    
    sim_violations[vis_sim] += num_violations
    artificial_violations[artificial] += num_violations
    sum_violations += num_violations
    
    # for comparison, also compute expected number of violations for randomly chosen points  
    avg_uniform_violations = 0
    avg_normal_violations = 0
    avg_shuffled_violations = 0

    num_repetitions = 100    
    
    for i in range(num_repetitions):
        uniform_hull_points = np.random.rand(len(hull_points), args.n)
        uniform_query_points = np.random.rand(len(query_points), args.n)
        avg_uniform_violations += count_violations(uniform_hull_points, uniform_query_points)       
        
        normal_hull_points = np.random.normal(size=(len(hull_points), args.n))
        normal_query_points = np.random.normal(size=(len(query_points), args.n))
        avg_normal_violations += count_violations(normal_hull_points, normal_query_points)       
        
        shuffled_data_points = np.array(list(vectors.values()))
        np.random.shuffle(shuffled_data_points)
        shuffled_hull_points = shuffled_data_points[:len(hull_points)]
        shuffled_query_points = shuffled_data_points[len(hull_points):]
        avg_shuffled_violations += count_violations(shuffled_hull_points, shuffled_query_points)  
    
    avg_uniform_violations /= num_repetitions
    uniform_sum_violations += avg_uniform_violations
    
    avg_normal_violations /= num_repetitions
    normal_sum_violations += avg_normal_violations
    
    avg_shuffled_violations /= num_repetitions
    shuffled_sum_violations += avg_shuffled_violations
        
    print("{0} ({1}, {2}): {3} violations (uniform: {4}, normal: {5}, shuffled: {6})".format(category, vis_sim, artificial, 
              num_violations, avg_uniform_violations, avg_normal_violations, avg_shuffled_violations))

print("Total violations: {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(sum_violations, uniform_sum_violations, 
                                                                                  normal_sum_violations, shuffled_sum_violations))
print("Classification with respect to Sim/Dis: {0}".format(sim_violations))
print("Classification with respect to Art/Nat: {0}".format(artificial_violations))