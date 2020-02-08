# -*- coding: utf-8 -*-
"""
Filters the interpretable directions (by taking the average of the best ones) and outputs a clean csv file.

Created on Thu Jan 16 11:10:45 2020

@author: lbechberger
"""

import argparse, os, csv, fcntl
import numpy as np
from code.util import normalize_direction

parser = argparse.ArgumentParser(description='Filtering interpretable directions')
parser.add_argument('input_file', help = 'the input csv file for the given direction')
parser.add_argument('direction_name', help = 'human-readable name for the direction')
parser.add_argument('n_dims', type = int, help = 'the maximal number of dimensions to consider')
parser.add_argument('output_file', help = 'output csv file for storing the identified directions')
parser.add_argument('-k', '--kappa_threshold', type = float, help = 'minimal kappa value needed to pass the filter', default = 0.0)
parser.add_argument('-s', '--spearman_threshold', type = float, help = 'minimal spearman correlation needed to pass the filter', default = 0.0)
args = parser.parse_args()

thresholds = {'kappa': args.kappa_threshold, 'spearman': args.spearman_threshold }

data_dict = {}
for i in range(1, args.n_dims + 1):
    data_dict[i] = []
    
with open(args.input_file, 'r') as f_in:            
    reader = csv.DictReader(f_in)
    for row in reader:
        dims = int(row['dims'])
        scale_type = row['scale_type']
        model = row['model']
        kappa = float(row['kappa'])
        spearman = float(row['spearman'])
        vector = np.array([float(row['d{0}'.format(d)]) for d in range(dims)])
        
        data_tuple = ('{0}-{1}'.format(scale_type, model), kappa, spearman, vector)
        data_dict[dims].append(data_tuple)

# create file and headline if necessary
if not os.path.exists(args.output_file):
    headline = 'dims,direction_name,criterion,constructed_from,{0}\n'.format(','.join(['d{0}'.format(i) for i in range(20)]))
    with open(args.output_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(headline)
        fcntl.flock(f, fcntl.LOCK_UN)

# helper function to aggregate direction based on the given criterion
def aggregate_direction(tuples, criterion, criterion_idx):
    max_value = max(map(lambda x: x[criterion_idx], tuples))
    if max_value < thresholds[criterion]:
        # best value did not exceed the given threshold: break
        return ([],[])
    max_tuples = [(x[0], x[3]) for x in tuples if x[criterion_idx] == max_value]
    constructed_from = ' '.join(map(lambda x: x[0], max_tuples))
    mean_direction = sum(map(lambda x: x[1], max_tuples)) /len(max_tuples)
    mean_direction = normalize_direction(mean_direction)    
    
    return (constructed_from, mean_direction)
    
# look at each similarity space separately
for dims in range(1, args.n_dims + 1):
    
    for criterion, criterion_idx in [('kappa', 1), ('spearman', 2)]:
        constructed_from, mean_direction = aggregate_direction(data_dict[dims], criterion, criterion_idx)

        if len(constructed_from) > 0:
            # only make output if we passed the filtering step
            with open(args.output_file, 'a') as f_out:
                line_items = [dims, args.direction_name, criterion, constructed_from] + mean_direction.tolist()
                f_out.write(','.join(map(lambda x: str(x), line_items)))
                f_out.write('\n')