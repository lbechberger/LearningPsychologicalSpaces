# -*- coding: utf-8 -*-
"""
Script to normalize all the MDS spaces in a given folder (overwrites the original files).

Created on Thu Feb  7 08:48:04 2019

@author: lbechberger
"""

import argparse, os
import numpy as np
from code.util import load_mds_vectors, normalize_vectors

parser = argparse.ArgumentParser(description='Normalizes all MDS spaces in the given folder')
parser.add_argument('input_folder', help = 'the directory containing all the original MDS spaces')
parser.add_argument('-b', '--backup', action = 'store_true', help = 'create a backup of the original spaces')
parser.add_argument('-v', '--verbose', action = 'store_true', help = 'enable verbose output for debug purposes')
args = parser.parse_args()

# look at all the files in the folder
for file_name in os.listdir(args.input_folder):
    
    path_to_file = os.path.join(args.input_folder, file_name)
    
    if os.path.isfile(path_to_file) and file_name.endswith('D-vectors.csv'):
        
        if args.verbose:
            print(file_name)
        
        dictionary = load_mds_vectors(path_to_file)
        
        # read all the vectors and the item names
        item_ids = list(sorted(dictionary.keys()))
        vectors = []
        for item_id in item_ids:
            vectors.append(dictionary[item_id])
        vectors = np.array(vectors)
        
        # normalize them
        normalized_vectors = normalize_vectors(vectors)
                
        # store the results in the file
        with open(path_to_file, 'w') as f:
            for item_id, vector in zip(item_ids, normalized_vectors):
                f.write('{0},{1}\n'.format(item_id, ','.join(map(lambda x: str(x), vector))))
        
        # create backup if necessary
        if args.backup:
            with open(path_to_file.replace('.csv', '-backup.csv'), 'w') as f:
                for item_id, vector in zip(item_ids, vectors):
                    f.write('{0},{1}\n'.format(item_id, ','.join(map(lambda x: str(x), vector))))

            