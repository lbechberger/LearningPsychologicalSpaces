# -*- coding: utf-8 -*-
"""
Analyze the overlap between conceptual regions.

Created on Wed Nov 14 10:57:29 2018

@author: lbechberger
"""

import pickle, argparse, os, fcntl
import numpy as np
from scipy.optimize import linprog
from code.util import load_mds_vectors

parser = argparse.ArgumentParser(description='Analyze overlap of conceptual regions')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
parser.add_argument('data_set_file', help = 'the pickle file containing the data set')
parser.add_argument('n', type = int, help = 'number of dimensions of the MDS space')
parser.add_argument('-o', '--output_file', help = 'output csv file for collecting the results', default = None)
parser.add_argument('-r', '--repetitions', type = int, help = 'number of repetitions in sampling the baselines', default = 20)
parser.add_argument('-b', '--baseline', action = "store_true", help = 'whether or not to compute the random baselines')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation when computing baselines', default = None)
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
vectors = load_mds_vectors(args.vector_file)

items = list(vectors.keys())
categories = []
for item in items:
    category = data_set['items'][item]['category']
    if category not in categories:
        categories.append(category)


# prepare the dictionaries for collecting the violation counts
def get_internal_dict():
    return {'MDS':0, 'uniform':0, 'normal':0, 'shuffled':0}

sum_violations = get_internal_dict()
sim_violations = {}
art_violations = {}

for this_type in ['Sim', 'Dis']:
    sim_violations[this_type] = {}
    for other_type in ['Sim', 'Dis']:
        sim_violations[this_type][other_type] = get_internal_dict()
    
for this_type in ['art', 'nat']:
    art_violations[this_type] = {}
    for other_type in ['art', 'nat']:
        art_violations[this_type][other_type] = get_internal_dict()

if args.seed is not None:
    np.random.seed(args.seed)

# iterate over all categories
for category_1 in categories:
    items_in_category_1 = [item for item in items if item in data_set['categories'][category_1]['items']]
            
    for category_2 in categories:
        items_in_category_2 = [item for item in items if item in data_set['categories'][category_2]['items']]
        
        # counts will anyways be zero, so skip
        if category_1 == category_2:
            continue
        
        # grab the corresponding points 
        hull_points = np.array([vectors[item] for item in items_in_category_1]) 
        query_points = [vectors[item] for item in items_in_category_2]    
        
        # count the number of violations: how many items of category 2 are inside category 1?
        num_violations = count_violations(hull_points, query_points)
        
        sim_1 = data_set['categories'][category_1]['visSim']
        sim_2 = data_set['categories'][category_2]['visSim']
        art_1 = data_set['categories'][category_1]['artificial']
        art_2 = data_set['categories'][category_2]['artificial']
        
        # add counts
        sim_violations[sim_1][sim_2]['MDS'] += num_violations
        art_violations[art_1][art_2]['MDS'] += num_violations
        sum_violations['MDS'] += num_violations
               
        if args.baseline:
            # for comparison, also compute expected number of violations for randomly chosen points  
            avg_uniform_violations = 0
            avg_normal_violations = 0
            avg_shuffled_violations = 0
        
            for i in range(args.repetitions):
                # UNIFORM
                uniform_hull_points = np.random.rand(len(hull_points), args.n)
                uniform_query_points = np.random.rand(len(query_points), args.n)
                avg_uniform_violations += count_violations(uniform_hull_points, uniform_query_points)       
                
                # NORMAL
                normal_hull_points = np.random.normal(size=(len(hull_points), args.n))
                normal_query_points = np.random.normal(size=(len(query_points), args.n))
                avg_normal_violations += count_violations(normal_hull_points, normal_query_points)       
                
                # SHUFFLED
                shuffled_data_points = np.array(list(vectors.values()))
                np.random.shuffle(shuffled_data_points)
                shuffled_hull_points = shuffled_data_points[:len(hull_points)]
                shuffled_query_points = shuffled_data_points[len(hull_points):len(hull_points)+len(query_points)]
                avg_shuffled_violations += count_violations(shuffled_hull_points, shuffled_query_points)  
            
            # UNIFORM
            avg_uniform_violations /= args.repetitions
            # add counts
            sim_violations[sim_1][sim_2]['uniform'] += avg_uniform_violations
            art_violations[art_1][art_2]['uniform'] += avg_uniform_violations
            sum_violations['uniform'] += avg_uniform_violations
        
            # NORMAL
            avg_normal_violations /= args.repetitions
            # add counts
            sim_violations[sim_1][sim_2]['normal'] += avg_normal_violations
            art_violations[art_1][art_2]['normal'] += avg_normal_violations
            sum_violations['normal'] += avg_normal_violations
            
            # SHUFFLED
            avg_shuffled_violations /= args.repetitions
            # add counts
            sim_violations[sim_1][sim_2]['shuffled'] += avg_shuffled_violations
            art_violations[art_1][art_2]['shuffled'] += avg_shuffled_violations
            sum_violations['shuffled'] += avg_shuffled_violations
               
            print("{0} ({1}, {2}) - {3} ({4}, {5}): {6} violations (uniform: {7}, normal: {8}, shuffled: {9})".format(category_1, sim_1, art_1, 
                      category_2, sim_2, art_2, num_violations, avg_uniform_violations, avg_normal_violations, avg_shuffled_violations))
        else:
            print("{0} ({1}, {2}) - {3} ({4}, {5}): {6} violations ".format(category_1, sim_1, art_1, category_2, sim_2, art_2, num_violations))


print("\nTotal violations: {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(sum_violations['MDS'], 
                                                                                  sum_violations['uniform'], 
                                                                                  sum_violations['normal'], 
                                                                                  sum_violations['shuffled']))
print("\nVisual Similarity:")
for sim_1 in ['Sim', 'Dis']:
    for sim_2 in ['Sim', 'Dis']:
        print("\t{0} - {1}: {2} (uniform: {3}, normal: {4}, shuffled: {5})".format(sim_1, sim_2,
                      sim_violations[sim_1][sim_2]['MDS'], sim_violations[sim_1][sim_2]['uniform'],
                        sim_violations[sim_1][sim_2]['normal'], sim_violations[sim_1][sim_2]['shuffled']))

print("\nNaturalness:")
for art_1 in ['art', 'nat']:
    for art_2 in ['art', 'nat']:
        print("\t\t{0} - {1}: {2} (uniform: {3}, normal: {4}, shuffled: {5})".format(art_1, art_2,
                      art_violations[art_1][art_2]['MDS'], art_violations[art_1][art_2]['uniform'],
                        art_violations[art_1][art_2]['normal'], art_violations[art_1][art_2]['shuffled']))


if args.output_file != None:
    
    # write headline if necessary
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            
            headline_items = ['dims']
            
            # total
            headline_items.append('total')
            if args.baseline:
                for distr in ['u', 'n', 's']:
                    headline_items.append('_'.join(['total', distr]))
        
            # sim
            for sim_1 in ['s', 'd']:
                for sim_2 in ['s', 'd']:
                    headline_items.append('_'.join(['sim', sim_1, sim_2]))
                    if args.baseline:
                        for distr in ['u', 'n', 's']:
                            headline_items.append('_'.join(['sim', sim_1, sim_2, distr]))
    
            # art
            for art_1 in ['a', 'n']:
                for art_2 in ['a', 'n']:
                    headline_items.append('_'.join(['art', art_1, art_2]))
                    if args.baseline:
                        for distr in ['u', 'n', 's']:
                            headline_items.append('_'.join(['art', art_1, art_2, distr]))
        
            f.write("{0}\n".format(','.join(headline_items)))
            fcntl.flock(f, fcntl.LOCK_UN)
            
    
    # write content
    with open(args.output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
            
        # total
        total = []
        total.append(sum_violations['MDS'])
        if args.baseline:
            for distribution in ['uniform', 'normal', 'shuffled']:
                total.append(sum_violations[distribution])
        
        # sim
        sim = []
        for sim_1 in ['Sim', 'Dis']:
            for sim_2 in ['Sim', 'Dis']:
                sim.append(sim_violations[sim_1][sim_2]['MDS'])
                if args.baseline:
                    for distribution in ['uniform', 'normal', 'shuffled']:
                        sim.append(sim_violations[sim_1][sim_2][distribution])
    
        # art
        art = []
        for art_1 in ['art', 'nat']:
            for art_2 in ['art', 'nat']:
                art.append(art_violations[art_1][art_2]['MDS'])
                if args.baseline:
                    for distribution in ['uniform', 'normal', 'shuffled']:
                        art.append(art_violations[art_1][art_2][distribution])
        
        list_of_all_results = [args.n] + total + sim + art  
        f.write(','.join(map(lambda x: str(x), list_of_all_results)))
        f.write('\n')
        fcntl.flock(f, fcntl.LOCK_UN)
            