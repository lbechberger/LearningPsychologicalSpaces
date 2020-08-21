# -*- coding: utf-8 -*-
"""
Analyze the size of the conceptual regions.

Created on Wed Jan 29 12:47:17 2020

@author: lbechberger
"""

import pickle, argparse, os, fcntl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from code.util import normalize_vectors

parser = argparse.ArgumentParser(description='Analyzing the size of conceptual regions')
parser.add_argument('input_file', help = 'the input pickle file containing the vectors and the category structure')
parser.add_argument('n_dims', type = int, help = 'dimensionality of space to investigate')
parser.add_argument('output_file', help = 'output csv file for collecting the results')
parser.add_argument('-b', '--baseline', action = "store_true", help = 'whether or not to compute the random baselines')
parser.add_argument('-r', '--repetitions', type = int, help = 'number of repetitions in sampling the baselines', default = 20)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation when computing baselines', default = None)
args = parser.parse_args()

# computes the mean Euclidean distance of the category points to their prototype
def mean_distance_to_prototype(category_points):
    prototype = np.mean(category_points, axis=0)
    distances = [euclidean_distances(point,prototype)[0][0] for point in category_points]
    return np.mean(distances)

# read the data set
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)
categories = sorted(data['categories'].keys())
vectors = data[args.n_dims]

if args.seed is not None:
    np.random.seed(args.seed)

# prepare dictionary for storing output
def get_internal_dict():
    return {'MDS':[], 'uniform':[], 'normal':[], 'shuffled':[]}

average_distances = {}
for category_type in ['VC', 'VV', 'all']:
    average_distances[category_type] = get_internal_dict()

# iterate over all categories
for category in categories:
    
    items_in_category = data['categories'][category]['items']
    category_points = []    
    for item in items_in_category:  
        vec = np.array(vectors[item])
        category_points.append(vec.reshape((-1,1)))
    #category_points = np.array([vectors[item] for item in items_in_category]) 
    mean_distance = mean_distance_to_prototype(category_points)
    
    avg_uniform_distance = 0
    avg_normal_distance = 0
    avg_shuffled_distance = 0
    
    if args.baseline:
        # for comparison, also compute expected distance to prototype for randomly chosen points  
        for i in range(args.repetitions):
            # UNIFORM
            uniform_points = np.random.rand(len(category_points), args.n_dims, 1)
            uniform_points = normalize_vectors(uniform_points)
            avg_uniform_distance += mean_distance_to_prototype(uniform_points)
            
            # NORMAL
            normal_points = np.random.normal(size=(len(category_points), args.n_dims, 1))
            normal_points = normalize_vectors(normal_points)
            avg_normal_distance += mean_distance_to_prototype(normal_points)
            
            # SHUFFLED
            shuffled_data_points = []
            for vec in vectors.values():
                np_vec = np.array(vec)
                shuffled_data_points.append(np_vec.reshape((-1,1)))
            np.random.shuffle(shuffled_data_points)
            shuffled_points = shuffled_data_points[:len(category_points)]
            avg_shuffled_distance += mean_distance_to_prototype(shuffled_points)
    
        avg_uniform_distance /= args.repetitions
        avg_normal_distance /= args.repetitions
        avg_shuffled_distance /= args.repetitions
    
         
    category_type = data['categories'][category]['visSim']
    for cat_type in [category_type, 'all']:
        average_distances[cat_type]['MDS'].append(mean_distance)
        average_distances[cat_type]['uniform'].append(avg_uniform_distance)
        average_distances[cat_type]['normal'].append(avg_normal_distance)
        average_distances[cat_type]['shuffled'].append(avg_shuffled_distance)


# write headline if necessary
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f_out:
        fcntl.flock(f_out, fcntl.LOCK_EX)
        
        headline_items = ['dims']
        
        for category_type in ['all', 'VC', 'VV']:
            headline_items.append(category_type)
            if args.baseline:
                for distr in ['u', 'n', 's']:
                    headline_items.append('_'.join([category_type, distr]))
               
        f_out.write("{0}\n".format(','.join(headline_items)))
        fcntl.flock(f_out, fcntl.LOCK_UN)
        

# write content
with open(args.output_file, 'a') as f_out:
    fcntl.flock(f_out, fcntl.LOCK_EX)
    
    output_line = [args.n_dims]
    for category_type in ['all', 'VC', 'VV']:
        output_line.append(np.mean(average_distances[category_type]['MDS']))
        if args.baseline:
            for distribution in ['uniform', 'normal', 'shuffled']:
                output_line.append(np.mean(average_distances[category_type][distribution]))

    f_out.write(','.join(map(lambda x: str(x), output_line)))
    f_out.write('\n')
    fcntl.flock(f_out, fcntl.LOCK_UN)
        