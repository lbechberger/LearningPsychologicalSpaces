# -*- coding: utf-8 -*-
"""
Shuffle the ground truth labels and store both the original and the shuffled form in a pickle file.

Created on Wed Feb  7 14:50:22 2018

@author: lbechberger
"""

import random, pickle, argparse
from code.util import load_mds_vectors

parser = argparse.ArgumentParser(description='Preparing the target vectors for the regression')
parser.add_argument('input_file', help = 'csv file containing the paths to target vector files')
parser.add_argument('output_file', help = 'pickle file for storing the results')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

if args.seed is None:
   args.seed = random.randint(0,1000)

result = {}

with open(args.input_file, 'r') as f:
    for line in f:
        
        tokens = line.replace('\n','').split(',')
        if len(tokens) != 2:
            continue
        target_name = tokens[0]
        target_file = tokens[1]

        real_dict = load_mds_vectors(target_file)
        shuffled_dict = {}
        
        # shuffle them
        keys = list(real_dict.keys())
        values = list(real_dict.values())
        # need to seed every time to make sure that the shuffled mapping is identical for all the different files
        random.seed(args.seed)
        random.shuffle(values)
        for i in range(len(keys)):
            shuffled_dict[keys[i]] = values[i]
        
        # store both original and shuffled ones
        result[target_name] = {'correct': real_dict, 'shuffled': shuffled_dict}

pickle.dump(result, open(args.output_file, 'wb'))