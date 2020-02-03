# -*- coding: utf-8 -*-
"""
Analyzes a given dimension based on its ratings and creates regression and classification targets for downstream tasks.

Created on Thu Jan  9 12:44:34 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from code.util import load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description='Analyzing dimension data')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed dimension data')
parser.add_argument('analysis_folder', help = 'folder where the plots will be stored')
parser.add_argument('classification_file', help = 'output pickle file for the classification information')
parser.add_argument('regression_file', help = 'output pickle file for the regression information')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-r', '--response_times', action = 'store_true', help = 'additionally convert response times into scale')
args = parser.parse_args()

# load dimension data
with open(args.input_file, 'rb') as f_in:
    dimension_data = pickle.load(f_in)

# sorted list of item_ids
items_sorted = list(sorted(list(dimension_data.keys())))
item_names = {}
for item in items_sorted:
    item_names[item] = dimension_data[item]['name']

# then read in all the images
images = None
if args.image_folder != None:
    images = load_item_images(args.image_folder, items_sorted)    


aggregated_binary = {}
aggregated_continuous_mean = {}
aggregated_continuous_median = {}

if args.response_times:
    aggregated_rt_mean = {}
    aggregated_rt_median = {}
    max_rt_mean = float('-inf')
    min_rt_mean = float('inf')
    max_rt_median = float('-inf')
    min_rt_median = float('inf')

# aggregate the individual responses into an overall score on a scale from -1 to 1
for item_id, inner_dict in dimension_data.items():
    # collect information about this item
    binary_responses = list(map(lambda x: x[1], inner_dict['binary']))
    binary_rts = list(map(lambda x: x[0], inner_dict['binary']))
    continuous = inner_dict['continuous']
    
    # aggregate classification decision into scale: percentage of True - percentage of False --> value between -1 and 1
    binary_value = (binary_responses.count(True) - binary_responses.count(False)) / len(binary_responses)
    aggregated_binary[item_id] = binary_value
    
    if args.response_times:
        # aggregate response time into scale: take median RT
        rt_mean = np.mean(binary_rts)
        # put through logarithm (RT ~ e^-x --> x ~ -ln RT)
        rt_abs_mean = -np.log(rt_mean)
        aggregated_rt_mean[item_id] = rt_abs_mean
        # keep maximum and minimum up to date
        max_rt_mean = max(max_rt_mean, rt_abs_mean)
        min_rt_mean = min(min_rt_mean, rt_abs_mean)
    
        rt_median = np.median(binary_rts)
        rt_abs_median = -np.log(rt_median)
        aggregated_rt_median[item_id] = rt_abs_median
        max_rt_median = max(max_rt_median, rt_abs_median)
        min_rt_median = min(min_rt_median, rt_abs_median)

    # aggregate continouous rating into scale: take median/median and rescale it from [0,1000] to [-1,1]
    continuous_mean = np.mean(continuous)
    continuous_mean_value = (continuous_mean / 500) - 1
    aggregated_continuous_mean[item_id] = continuous_mean_value

    continuous_median = np.median(continuous)
    continuous_median_value = (continuous_median / 500) - 1
    aggregated_continuous_median[item_id] = continuous_median_value

if args.response_times:
    # need to rescale the RT-based ratings onto a scale between -1 and 1
    for item_id, rt_abs in aggregated_rt_mean.items():
        rt_abs_rescaled = (rt_abs - min_rt_mean) / (max_rt_mean - min_rt_mean)
        # multiply with sign of binary_value to distinguish positive from negative examples
        rt_value = rt_abs_rescaled * np.sign(aggregated_binary[item_id])
        aggregated_rt_mean[item_id] = rt_value
        
    for item_id, rt_abs in aggregated_rt_median.items():
        rt_abs_rescaled = (rt_abs - min_rt_median) / (max_rt_median - min_rt_median)
        # multiply with sign of binary_value to distinguish positive from negative examples
        rt_value = rt_abs_rescaled * np.sign(aggregated_binary[item_id])
        aggregated_rt_median[item_id] = rt_value
    
    
# store this information as regression output
regression_output = {'binary': aggregated_binary, 'continuous_mean': aggregated_continuous_mean, 'continuous_median': aggregated_continuous_median}
if args.response_times:
    regression_output['rt_mean'] = aggregated_rt_mean
    regression_output['rt_median'] = aggregated_rt_median
    
with open(args.regression_file, 'wb') as f_out:
    pickle.dump(regression_output, f_out)    

# look at all pairs of scales
for first_scale, second_scale in combinations(sorted(regression_output.keys()), 2):
    
    # compute Spearman correlation
    first_vector = []
    second_vector = []
    for item_id in items_sorted:
        first_vector.append(regression_output[first_scale][item_id])
        second_vector.append(regression_output[second_scale][item_id])
    spearman, _ = spearmanr(first_vector, second_vector)
    print("Spearman correlation between {0} and {1}: {2}".format(first_scale, second_scale, spearman))
    
    # create scatter plot
    output_file_name = os.path.join(args.analysis_folder, 'scatter-{0}-{1}.png'.format(first_scale, second_scale))        
    create_labeled_scatter_plot(first_vector, second_vector, output_file_name, x_label = first_scale, y_label = second_scale, images = images, zoom = args.zoom)  

# now binarize the scales by taking the top 25% and bottom 25% as positive and negative examples, respectively
classification_output = {}
for scale_name, scale_data in regression_output.items():
    list_of_tuples = []
    for item_id, value in scale_data.items():
        list_of_tuples.append((item_id, value))
    list_of_tuples = sorted(list_of_tuples, key = lambda x: x[1])
    
    negative_cutoff = int(len(list_of_tuples) / 4)
    positive_cutoff = len(list_of_tuples) - negative_cutoff
    positive = list(map(lambda x: x[0], list_of_tuples[positive_cutoff:]))
    negative = list(map(lambda x: x[0], list_of_tuples[:negative_cutoff]))    
    
    classification_output[scale_name] = {'positive': positive, 'negative': negative}
    
    print('Positive examples for {0}'.format(scale_name))
    print(','.join(sorted(map(lambda x: item_names[x], positive))))
    print('Negative examples for {0}'.format(scale_name))
    print(','.join(sorted(map(lambda x: item_names[x], negative))))

# store this information as classification output
with open(args.classification_file, 'wb') as f_out:
    pickle.dump(classification_output, f_out)    

# look at all pairs of scales
for first_scale, second_scale in combinations(sorted(classification_output.keys()), 2):

    first_data = classification_output[first_scale]
    second_data = classification_output[second_scale]
    
    intersection_pos = len(set(first_data['positive']).intersection(second_data['positive']))
    intersection_neg = len(set(first_data['negative']).intersection(second_data['negative']))
    
    print('Comparing classification of {0} to {1}: Intersection Pos {2}, Intersection Neg {3}'.format(first_scale, second_scale, intersection_pos, intersection_neg))