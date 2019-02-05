# -*- coding: utf-8 -*-
"""
Little helper script for storing the dissimilarity matrix from the pickle file 
in the form of a csv file in order to use with R.

Created on Thu Jan 31 12:12:01 2019

@author: lbechberger
"""
import os, pickle, argparse

parser = argparse.ArgumentParser(description='Creating CSV files for R code')
parser.add_argument('input_file', help = 'the input file to use')
parser.add_argument('output_folder', help = 'the output folder to use')
args = parser.parse_args()

with open(args.input_file, 'rb') as f:
    input_data = pickle.load(f)

items_of_interest = input_data['items']
dissimilarity_matrix = input_data['dissimilarities']

with open(os.path.join(args.output_folder, 'distance_matrix.csv'), 'w') as f:
    for line in dissimilarity_matrix:
        f.write('{0}\n'.format(','.join(map(lambda x: str(x), line))))

with open(os.path.join(args.output_folder, 'item_names.csv'), 'w') as f:
    for item in items_of_interest:
        f.write("{0}\n".format(item))