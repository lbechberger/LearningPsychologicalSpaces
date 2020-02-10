# -*- coding: utf-8 -*-
"""
Analyzes a given pychological feature based on its ratings and creates regression and classification targets for downstream tasks.

Created on Thu Jan  9 12:44:34 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from code.util import load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description='Analyzing psychological feature')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed feature data')
parser.add_argument('analysis_folder', help = 'folder where the plots will be stored')
parser.add_argument('classification_file', help = 'output pickle file for the classification information')
parser.add_argument('regression_file', help = 'output pickle file for the regression information')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-r', '--response_times', action = 'store_true', help = 'additionally consider response times')
parser.add_argument('-m', '--median', action = 'store_true', help = 'use the median for aggregation instead of the mean')
args = parser.parse_args()

# set aggregator function based on parameter
aggregator = np.median if args.median else np.mean

# load feature data
with open(args.input_file, 'rb') as f_in:
    feature_data = pickle.load(f_in)

# sorted list of item_ids
items_sorted = list(sorted(list(feature_data.keys())))
item_names = {}
for item in items_sorted:
    item_names[item] = feature_data[item]['name']

# then read in all the images
images = None
if args.image_folder != None:
    images = load_item_images(args.image_folder, items_sorted)    

aggregated_pre_attentive = {}
aggregated_attentive = {}

if args.response_times:
    aggregated_rt = {}
    max_rt = float('-inf')
    min_rt = float('inf')

# aggregate the individual responses into an overall score on a scale from -1 to 1
for item_id, inner_dict in feature_data.items():
    # collect information about this item
    pre_attentive_responses = list(map(lambda x: x[1], inner_dict['pre-attentive']))
    attentive_responses = inner_dict['attentive']
    
    # aggregate pre-attentive classification decision into scale: percentage of True - percentage of False --> value between -1 and 1
    pre_attentive_value = (pre_attentive_responses.count(True) - pre_attentive_responses.count(False)) / len(pre_attentive_responses)
    aggregated_pre_attentive[item_id] = pre_attentive_value
    
    if args.response_times:
        # load RTs
        rts = list(map(lambda x: x[0], inner_dict['pre-attentive']))
        # aggregate response time into scale: take mean/median RT
        rt = aggregator(rts)
        # put through logarithm (RT ~ e^-x --> x ~ -ln RT)
        rt_abs = -np.log(rt)
        aggregated_rt[item_id] = rt_abs
        # keep maximum and minimum up to date
        max_rt = max(max_rt, rt_abs)
        min_rt = min(min_rt, rt_abs)
    
    # aggregate attentive continouous rating into scale: take mean/median and rescale it from [0,1000] to [-1,1]
    attentive = aggregator(attentive_responses)
    attentive_value = (attentive / 500) - 1
    aggregated_attentive[item_id] = attentive_value

if args.response_times:
    # need to rescale the RT-based ratings onto a scale between -1 and 1
    for item_id, rt_abs in aggregated_rt.items():
        rt_abs_rescaled = (rt_abs - min_rt) / (max_rt - min_rt)
        # multiply with sign of binary_value to distinguish positive from negative examples
        rt_value = rt_abs_rescaled * np.sign(aggregated_attentive[item_id])
        aggregated_rt[item_id] = rt_value
    
# store this information as regression output
regression_output = {'pre-attentive': aggregated_pre_attentive, 'attentive': aggregated_attentive}
if args.response_times:
    regression_output['rt'] = aggregated_rt
    
with open(args.regression_file, 'wb') as f_out:
    pickle.dump(regression_output, f_out)    

# look at all pairs of feature types
for first_feature_type, second_feature_type in combinations(sorted(regression_output.keys()), 2):
    
    # compute Spearman correlation
    first_vector = []
    second_vector = []
    for item_id in items_sorted:
        first_vector.append(regression_output[first_feature_type][item_id])
        second_vector.append(regression_output[second_feature_type][item_id])
    spearman, _ = spearmanr(first_vector, second_vector)
    print("Spearman correlation between {0} and {1}: {2}".format(first_feature_type, second_feature_type, spearman))
    
    # create scatter plot
    output_file_name = os.path.join(args.analysis_folder, 'scatter-{0}-{1}.png'.format(first_feature_type, second_feature_type))        
    create_labeled_scatter_plot(first_vector, second_vector, output_file_name, x_label = first_feature_type, y_label = second_feature_type, images = images, zoom = args.zoom)  

# now binarize the scales by taking the top 25% and bottom 25% as positive and negative examples, respectively
classification_output = {}
for feature_type, feature_data in regression_output.items():
    list_of_tuples = []
    for item_id, value in feature_data.items():
        list_of_tuples.append((item_id, value))
    list_of_tuples = sorted(list_of_tuples, key = lambda x: x[1])
    
    negative_cutoff = int(len(list_of_tuples) / 4)
    positive_cutoff = len(list_of_tuples) - negative_cutoff
    positive = list(map(lambda x: x[0], list_of_tuples[positive_cutoff:]))
    negative = list(map(lambda x: x[0], list_of_tuples[:negative_cutoff]))    
    
    classification_output[feature_type] = {'positive': positive, 'negative': negative}
    
    print('Positive examples for {0}'.format(feature_type))
    print(','.join(sorted(map(lambda x: item_names[x], positive))))
    print('Negative examples for {0}'.format(feature_type))
    print(','.join(sorted(map(lambda x: item_names[x], negative))))

# store this information as classification output
with open(args.classification_file, 'wb') as f_out:
    pickle.dump(classification_output, f_out)    

# look at all pairs of scales
for first_feature_type, second_feature_type in combinations(sorted(classification_output.keys()), 2):

    first_data = classification_output[first_feature_type]
    second_data = classification_output[second_feature_type]
    
    intersection_pos = len(set(first_data['positive']).intersection(second_data['positive']))
    intersection_neg = len(set(first_data['negative']).intersection(second_data['negative']))
    
    print('Comparing classification of {0} to {1}: Intersection Pos {2}, Intersection Neg {3}'.format(first_feature_type, second_feature_type, intersection_pos, intersection_neg))