# -*- coding: utf-8 -*-
"""
Analyzes a given dimension based on its ratings and creates regression and classification targets for downstream tasks.

Created on Thu Jan  9 12:44:34 2020

@author: lbechberger
"""

import pickle, argparse

parser = argparse.ArgumentParser(description='Analyzing dimension data')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed dimension data')
parser.add_argument('analysis_folder', help = 'folder where the plots will be stored')
parser.add_argument('classification_file', help = 'output pickle file for the classification information')
parser.add_argument('regression_file', help = 'output pickle file for the regression information')
args = parser.parse_args()

with open(args.input_file, 'rb') as f_in:
    dimension_data = pickle.load(f_in)

# TODO for each item: aggregate classification decision (percentage), response time (median), and continuous rating (median)

# TODO translate the aggregated numbers into a corresponding scale 

# TODO compute Spearman correlation between the three scales

# TODO create scatter plots between the three scales

# TODO write regression output

# TODO take top and bottom quarter of each scale

# TODO compare the three sets with the Jaccard index (pairwise)

# write classification output