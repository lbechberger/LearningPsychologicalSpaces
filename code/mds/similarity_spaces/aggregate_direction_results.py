# -*- coding: utf-8 -*-
"""
Aggregate the evaluation results of the ML tasks for finding interpretable directions in order to make
further analysis easier

Created on Fri Feb  7 07:50:12 2020

@author: lbechberger
"""

import argparse, os, csv
import numpy as np

parser = argparse.ArgumentParser(description='Aggregating evaluation results for interpretable directions')
parser.add_argument('input_folder', help = 'the folder containing all input files')
parser.add_argument('n_dims', type = int, help = 'the maximal number of dimensions to consider')
parser.add_argument('output_folder', help = 'folder where the output csv files will be stored')
args = parser.parse_args()

# one entry for each dimension; then split up into 'feature_name', 'feature_type', and 'model'
# for each of them contain list of kappas and list of spearmans
direction_data = {}
for dims in range(1, args.n_dims + 1):
    direction_data[dims] = {'feature_name': {}, 'feature_type': {}, 'model': {}}

# create inner dictionary if necessary, store evaluation information
def add_to_dict(dictionary, key, kappa, spearman):
    if key not in dictionary:
        dictionary[key] = {'kappa': [], 'spearman': []}
    dictionary[key]['kappa'].append(kappa)
    dictionary[key]['spearman'].append(spearman)

# load direction data
for file_name in os.listdir(args.input_folder):
    if file_name.endswith('.csv'):
        feature_name = file_name.split('.')[0]
        for dims in range(1, args.n_dims + 1):
            direction_data[dims][feature_name] = {}

        with open(os.path.join(args.input_folder, file_name), 'r') as f_in:            
            reader = csv.DictReader(f_in)
            for row in reader:
                dims = int(row['dims'])
                feature_type = row['feature_type']
                model = row['model']
                kappa = float(row['kappa'])
                spearman = float(row['spearman'])
                
                add_to_dict(direction_data[dims]['feature_name'], feature_name, kappa, spearman)
                if feature_type != 'metadata':
                    # ignore the category-based directions when analyzing feature type and model
                    add_to_dict(direction_data[dims]['feature_type'], feature_type, kappa, spearman)
                    add_to_dict(direction_data[dims]['model'], model, kappa, spearman)

output = {'feature_name': [], 'feature_type': [], 'model': []}   
             
# iterate over all dimensions
for dims in range(1, args.n_dims + 1):
    dims_dict = direction_data[dims]
    
    for target in sorted(output.keys()):
        
        if len(output[target]) == 0:
            # need to construct headline
            headline = ['dims']
        
        line = [dims]
        
        # collect individual numbers
        for key in sorted(dims_dict[target].keys()):
            
            if len(output[target]) == 0:
                headline.append('{0}_kappa'.format(key))
                headline.append('{0}_spearman'.format(key))
                
            line.append(np.mean(dims_dict[target][key]['kappa']))
            line.append(np.mean(dims_dict[target][key]['spearman']))
        
        if len(output[target]) == 0:
            output[target].append(headline)
        output[target].append(line)

# now write the outputs
for target in sorted(output.keys()):
    output_file_name = os.path.join(args.output_folder, '{0}.csv'.format(target))
    with open(output_file_name, 'w') as f_out:
        for line in output[target]:
            f_out.write(','.join(map(lambda x: str(x), line)))
            f_out.write('\n')