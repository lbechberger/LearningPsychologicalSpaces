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
aggregated_rt = {}
aggregated_continuous = {}

# aggregate the individual responses into an overall score on a scale from -1 to 1
for item_id, inner_dict in dimension_data.items():
    # collect information about this item
    binary_responses = list(map(lambda x: x[1], inner_dict['binary']))
    binary_rts = list(map(lambda x: x[0], inner_dict['binary']))
    continuous = inner_dict['continuous']
    
    # aggregate classification decision into scale: percentage of True - percentage of False --> value between -1 and 1
    binary_value = (binary_responses.count(True) - binary_responses.count(False)) / len(binary_responses)
    aggregated_binary[item_id] = binary_value
    
    # aggregate response time into scale: take median RT, , multiply
    rt_median = np.median(binary_rts) / 1000
    # put through exponentially decaying function (RT = 0 gives value of 1, large RT gives value close to 0) --> how far away from decision surface
    rt_abs = np.exp(-rt_median)
    # multiply with sign of binary_value to distinguish positive from negative examples
    rt_value = rt_abs * np.sign(binary_value)
    aggregated_rt[item_id] = rt_value
    
    # aggregate continouous rating into scale: take median and rescale it from [0,1000] to [-1,1]
    continuous_median = np.median(continuous)
    continuous_value = (continuous_median / 500) - 1
    aggregated_continuous[item_id] = continuous_value
    
# store this information as regression output
regression_output = {'binary': aggregated_binary, 'rt': aggregated_rt, 'continuous': aggregated_continuous}
with open(args.regression_file, 'wb') as f_out:
    pickle.dump(regression_output, f_out)    

# look at all pairs of scales
for first_scale, second_scale in combinations(regression_output.keys(), 2):
    
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
    print(','.join(map(lambda x: item_names[x], positive)))
    print('Negative examples for {0}'.format(scale_name))
    print(','.join(map(lambda x: item_names[x], negative)))

# store this information as classification output
with open(args.classification_file, 'wb') as f_out:
    pickle.dump(classification_output, f_out)    

# look at all pairs of scales
for first_scale, second_scale in combinations(classification_output.keys(), 2):

    first_data = classification_output[first_scale]
    second_data = classification_output[second_scale]
    
    # helper function to compute jaccard index: intersection over unification
    def jaccard_index(list1, list2):
        intersection_size = len(set(list1).intersection(list2))
        unification_size = len(set(list1).union(list2))
        return intersection_size / unification_size

    jaccard_pos = jaccard_index(first_data['positive'], second_data['positive'])
    jaccard_neg = jaccard_index(first_data['negative'], second_data['negative'])
    
    print('Comparing classification of {0} to {1}: Jaccard Pos {2}, Jaccard Neg {3}'.format(first_scale, second_scale, jaccard_pos, jaccard_neg))