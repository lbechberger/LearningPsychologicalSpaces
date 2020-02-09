# -*- coding: utf-8 -*-
"""
Create features based on category information.

Created on Mon Feb  3 09:34:20 2020

@author: lbechberger
"""

import pickle, argparse, os
from code.util import select_data_subset

parser = argparse.ArgumentParser(description='Creating features from metadata about categories')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('regression_folder', help = 'path to the output folder for the regression data')
parser.add_argument('classification_folder', help = 'path to the output folder for the classification data')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
args = parser.parse_args()

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

# select subset of overall data set
items_of_interest, _, categories_of_interest =  select_data_subset(args.subset, data_set) 

candidate_features = [('visSim', 'Sim', 'Dis'), ('artificial', 'nat', 'art')]

for feature_name, neg_name, pos_name in candidate_features:
    pos_examples = []
    neg_examples = []


    for category in categories_of_interest:
        category_info = data_set['categories'][category]
        pos_examples += [item for item in category_info['items'] if category_info[feature_name] == pos_name and item in items_of_interest]
        neg_examples += [item for item in category_info['items'] if category_info[feature_name] == neg_name and item in items_of_interest]
        
    classification_output = {'metadata': {'positive': pos_examples, 'negative': neg_examples}}      
        
    with open(os.path.join(args.classification_folder, '{0}.pickle'.format(feature_name)), 'wb') as f_out:
        pickle.dump(classification_output, f_out)
    
    regression_map = {}
    for item in pos_examples:
        regression_map[item] = 1
    for item in neg_examples:
        regression_map[item] = -1
    regression_output = {'metadata': regression_map}
    
    with open(os.path.join(args.regression_folder, '{0}.pickle'.format(feature_name)), 'wb') as f_out:
        pickle.dump(regression_output, f_out)