# -*- coding: utf-8 -*-
"""
Script to normalize all the MDS spaces in a given folder (overwrites the original files).

Created on Thu Feb  7 08:48:04 2019

@author: lbechberger
"""

import argparse, os
import numpy as np

parser = argparse.ArgumentParser(description='Normalizes all MDS spaces in the given folder')
parser.add_argument('input_folder', help = 'the directory containing all the original MDS spaces')
parser.add_argument('-b', '--backup', action = 'store_true', help = 'create a backup of the original spaces')
parser.add_argument('-v', '--verbose', action = 'store_true', help = 'enable verbose output for debug purposes')
args = parser.parse_args()

# compute centroid of vectors
def centroid(vectors):
    return np.average(vectors, axis = 0)
         

# compute root mean squared distance of centered points to origin
def root_mean_squared_distance(vectors):
    squared_distances = np.sum(np.square(vectors), axis = 1)
    mean_squared_distance = np.average(squared_distances)        
    return np.sqrt(mean_squared_distance) 

# look at all the files in the folder
for file_name in os.listdir(args.input_folder):
    
    path_to_file = os.path.join(args.input_folder, file_name)
    
    if os.path.isfile(path_to_file) and file_name.endswith('.csv'):
        
        # read all the vectors and the item names
        item_ids = []        
        vectors = []
        with open(path_to_file, 'r') as f:
            for line in f:
                tokens = line.split(',')
                item_ids.append(tokens[0])
                vectors.append(list(map(lambda x: float(x), tokens[1:])))
        vectors = np.array(vectors)
        
        # make sure that centorid of vectors is at origin
        centered_vectors = vectors - centroid(vectors)
        
        # make sure that this root mean squared distance equals one
        normalized_vectors = centered_vectors / root_mean_squared_distance(centered_vectors)
        
        # output some debug information if necessary
        if args.verbose:
            print(file_name)
            print('before: centroid [{0}], RMSD {1}'.format(','.join(map(lambda x: str(x), centroid(vectors))), 
                                                              root_mean_squared_distance(centered_vectors)))
            print('after: centroid [{0}], RMSD {1}'.format(','.join(map(lambda x: str(x), centroid(normalized_vectors))), 
                                                              root_mean_squared_distance(normalized_vectors)))
            print('')
        
        # store the results in the file
        with open(path_to_file, 'w') as f:
            for item_id, vector in zip(item_ids, normalized_vectors):
                f.write('{0},{1}\n'.format(item_id, ','.join(map(lambda x: str(x), vector))))
        
        # create backup if necessary
        if args.backup:
            with open(path_to_file.replace('.csv', '-backup.csv'), 'w') as f:
                for item_id, vector in zip(item_ids, vectors):
                    f.write('{0},{1}\n'.format(item_id, ','.join(map(lambda x: str(x), vector))))

            