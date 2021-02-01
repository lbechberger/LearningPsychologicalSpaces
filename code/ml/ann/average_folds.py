# -*- coding: utf-8 -*-
"""
Averages the results per fold over all folds and stores them in a new csv file.

Created on Mon Feb  1 14:54:11 2021

@author: lbechberger
"""

import csv, argparse
import numpy as np

parser = argparse.ArgumentParser(description='Average fold results')
parser.add_argument('input_file', help = 'csv file containing the individual fold results')
parser.add_argument('output_file', help = 'csv file for the aggregated results')
args = parser.parse_args()


headline = []
content = {}

with open(args.input_file, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    headline = [col for col in reader.fieldnames if col not in ['configuration', 'fold']]
    
    for row in reader:
        config = row['configuration']
        if config not in content:
            content[config] = {}
            for col in headline:
                content[config][col] = []
        
        for col in headline:
            content[config][col].append(float(row[col]))

# write the results        
with open(args.output_file, 'w') as f_out:
    f_out.write('configuration,')
    f_out.write(','.join(headline))
    f_out.write('\n')
    
    for config, conf_dict in content.items():
        f_out.write('{0},'.format(config))
        averages = []
        for col in headline:
            averages.append(str(np.mean(conf_dict[col])))
        f_out.write(','.join(averages))
        f_out.write('\n')