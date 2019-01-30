# -*- coding: utf-8 -*-
"""
Analyze a given space for convextiy.

Created on Wed Nov 14 10:57:29 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser(description='Convexity analysis')
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

sim_violations = {'Sim' : {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}, 
                  'Dis' : {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}, 
                  'x' : {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}}
artificial_violations = {'art': {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}, 
                         'nat': {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}}
sum_violations = {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}

for category in categories:
    items_in_category = [item for item in items if item in data_set['categories'][category]['items']]
    items_not_in_category = [item for item in items if item not in items_in_category]
    
    # grab the corresponding points 
    hull_points = np.array([vectors[item] for item in items_in_category]) 
    query_points = [vectors[item] for item in items_not_in_category]    
    
    num_violations = count_violations(hull_points, query_points)
    
    vis_sim = data_set['categories'][category]['visSim']
    artificial = data_set['categories'][category]['artificial']
    
    sim_violations[vis_sim]['MDS'] += num_violations
    artificial_violations[artificial]['MDS'] += num_violations
    sum_violations['MDS'] += num_violations
    
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
    sim_violations[vis_sim]['uniform'] += avg_uniform_violations
    artificial_violations[artificial]['uniform'] += avg_uniform_violations
    sum_violations['uniform'] += avg_uniform_violations
    
    avg_normal_violations /= num_repetitions
    sim_violations[vis_sim]['normal'] += avg_normal_violations
    artificial_violations[artificial]['normal'] += avg_normal_violations
    sum_violations['normal'] += avg_normal_violations
    
    avg_shuffled_violations /= num_repetitions
    sim_violations[vis_sim]['shuffled'] += avg_shuffled_violations
    artificial_violations[artificial]['shuffled'] += avg_shuffled_violations
    sum_violations['shuffled'] += avg_shuffled_violations
        
    print("{0} ({1}, {2}): {3} violations (uniform: {4}, normal: {5}, shuffled: {6})".format(category, vis_sim, artificial, 
              num_violations, avg_uniform_violations, avg_normal_violations, avg_shuffled_violations))

print("\nTotal violations: {0} (uniform: {1}, normal: {2}, shuffled: {3})\n".format(sum_violations['MDS'], sum_violations['uniform'], 
                                                                                  sum_violations['normal'], sum_violations['shuffled']))

print("Within 'Sim': {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(sim_violations['Sim']['MDS'], sim_violations['Sim']['uniform'],
                                                                              sim_violations['Sim']['normal'], sim_violations['Sim']['shuffled']))
print("Within 'Dis': {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(sim_violations['Dis']['MDS'], sim_violations['Dis']['uniform'],
                                                                              sim_violations['Dis']['normal'], sim_violations['Dis']['shuffled']))
print("Within 'x': {0} (uniform: {1}, normal: {2}, shuffled: {3})\n".format(sim_violations['x']['MDS'], sim_violations['x']['uniform'],
                                                                              sim_violations['x']['normal'], sim_violations['x']['shuffled']))

print("Within 'art': {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(artificial_violations['art']['MDS'], artificial_violations['art']['uniform'],
                                                                              artificial_violations['art']['normal'], artificial_violations['art']['shuffled']))
print("Within 'nat': {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(artificial_violations['nat']['MDS'], artificial_violations['nat']['uniform'],
                                                                              artificial_violations['nat']['normal'], artificial_violations['nat']['shuffled']))
