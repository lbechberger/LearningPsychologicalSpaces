# -*- coding: utf-8 -*-
"""
Filters the interpretable directions according to given criteria and outputs a clean csv file.

Created on Thu Jan 16 11:10:45 2020

@author: lbechberger
"""

import argparse

parser = argparse.ArgumentParser(description='Filtering interpretable directions')
parser.add_argument('input_file', help = 'the input csv file containing the candidate directions')
parser.add_argument('direction_name', help = 'human-readable name for the direction')
parser.add_argument('output_file', help = 'output csv file for storing the identified directions')
parser.add_argument('-t', '--threshold', type = float, help = 'the threshold to use on the metric', default = 0.5)
parser.add_argument('-d', '--dataset', help = 'type of dataset to use', default = None)
args = parser.parse_args()


with open(args.input_file, 'r') as f_in:
    
    for line in f_in:
        if line.startswith('dims,dataset'):
            continue
    
    #TODO