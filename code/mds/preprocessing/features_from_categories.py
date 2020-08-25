# -*- coding: utf-8 -*-
"""
Create features based on category information.

Created on Mon Feb  3 09:34:20 2020

@author: lbechberger
"""

import pickle, argparse, os

parser = argparse.ArgumentParser(description='Creating features from metadata about categories')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('output_folder', help = 'path to the output folder for the feature data')
args = parser.parse_args()

# load the data set from the pickle file
with open(args.input_file, "rb") as f_in:
    data_set = pickle.load(f_in)

candidate_features = [('visSim', 'VC', 'VV'), ('artificial', 'nat', 'art')]

for feature_name, neg_name, pos_name in candidate_features:
    pos_examples = []
    neg_examples = []


    for category in data_set['category_names']:
        category_info = data_set['categories'][category]
        pos_examples += [item for item in category_info['items'] if category_info[feature_name] == pos_name]
        neg_examples += [item for item in category_info['items'] if category_info[feature_name] == neg_name]
        
    classification_output = {'metadata': {'positive': pos_examples, 'negative': neg_examples}}      
        

    regression_map = {}
    for item in pos_examples:
        regression_map[item] = 1
    for item in neg_examples:
        regression_map[item] = -1
    
    regression_output = {'metadata': regression_map}

    # no 'individual', since we don't have individual ratings here
    overall_output = {'aggregated': regression_output, 'classification': classification_output}

    with open(os.path.join(args.output_folder, '{0}.pickle'.format(feature_name)), 'wb') as f_out:
        pickle.dump(overall_output, f_out)
   
