# -*- coding: utf-8 -*-
"""
Script to normalize all the MDS spaces in a given folder (overwrites the original files).

Created on Thu Feb  7 08:48:04 2019

@author: lbechberger
"""

import argparse, os, pickle
import numpy as np
from code.util import load_mds_vectors, normalize_vectors

parser = argparse.ArgumentParser(description='Normalizes all MDS spaces in the given folder')
parser.add_argument('input_folder', help = 'the directory containing all the original MDS spaces')
parser.add_argument('input_file', help = 'path to the pickle file containing the category structure')
parser.add_argument('output_file', help = 'path to the output pickle file')
parser.add_argument('-b', '--backup', action = 'store_true', help = 'create a backup of the original spaces')
parser.add_argument('-v', '--verbose', action = 'store_true', help = 'enable verbose output for debug purposes')
args = parser.parse_args()

# read category information
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

output = {'categories': data['categories']}


# look at all the files in the folder
for file_name in os.listdir(args.input_folder):
    
    path_to_file = os.path.join(args.input_folder, file_name)
    
    if os.path.isfile(path_to_file) and file_name.endswith('D-vectors.csv'):
        
        if args.verbose:
            print(file_name)
        
        dictionary = load_mds_vectors(path_to_file)
        
        # extract n_dims from the file name
        n_dims = int(file_name.split('D')[0])
        
        # read all the vectors and the item names
        items = list(sorted(dictionary.keys()))
        vectors = []
        for item_id in items:
            vectors.append(dictionary[item_id])
        vectors = np.array(vectors)
        
        # normalize them
        normalized_vectors = normalize_vectors(vectors)
                
        # store the results in the file
        with open(path_to_file, 'w') as f_out:
            for item, vector in zip(items, normalized_vectors):
                f_out.write('{0},{1}\n'.format(item, ','.join(map(lambda x: str(x), vector))))
        
        # create backup if necessary
        if args.backup:
            with open(path_to_file.replace('.csv', '-backup.csv'), 'w') as f_out:
                for item, vector in zip(items, vectors):
                    f_out.write('{0},{1}\n'.format(item, ','.join(map(lambda x: str(x), vector))))
        
        # add to output for pickle file
        normalized_vector_dict = {}
        for item, vector in zip(items, vector):
            normalized_vector_dict[item] = vector
        output[n_dims] = normalized_vector_dict

# store pickle file
with open(args.output_file, 'wb') as f_out:
    pickle.dump(output, f_out)