# -*- coding: utf-8 -*-
"""
Read in the ratings about a given psychological feature.

Created on Thu Jan  9 11:48:19 2020

@author: lbechberger
"""

import pickle, argparse, csv, os
import numpy as np
from itertools import combinations
from code.util import load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description='Preprocessing ratings for psychological features')
parser.add_argument('pre_attentive_file', help = 'CSV file containing the pre-attentive feature ratings')
parser.add_argument('attentive_file', help = 'CSV file containing the attentive feature ratings')
parser.add_argument('categories_file', help = 'CSV file containing an ordered list of categories and their desired names')
parser.add_argument('items_file', help = 'CSV file containing a mapping of item names')
parser.add_argument('output_pickle_file', help = 'path to the output pickle file')
parser.add_argument('output_csv_file_individual', help = 'path to the output csv file for individual ratings')
parser.add_argument('output_csv_file_aggregated', help = 'path to the output csv file for aggregated ratings')
parser.add_argument('-p', '--plot_folder', help = 'folder where the plots shall be stored', default = None)
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
args = parser.parse_args()

response_mapping = {'keineAhnung': 0,
                    'länglich': -1, 'gleich': 1, 
                    'gebogen': 1, 'gerade': -1,
                    'vertikal': 1, 'horizontal': -1, 'diagonal1': 0, 'diagonal2': 0}

item_map_id_to_english = {}
item_map_english_to_id = {}
item_map_german_to_english = {}
item_names = []
category_map = {}
individual_ratings = {'pre-attentive':{}, 'attentive':{}}
aggregated_ratings = {'pre-attentive':{}, 'attentive':{}}

# read in the category names
with open(args.categories_file, 'r') as f:
    for line in f:
        tokens = line.replace('\n','').split(',')
        
        category_map[tokens[0]] = tokens[1]

# read in the item names
with open(args.items_file, 'r') as f:
    for line in f:
        tokens = line.replace('\n','').split(',')
        item_map_id_to_english[tokens[0]] = tokens[1]
        item_map_english_to_id[tokens[1]] = tokens[0]
        

# read in information from pre-attentive ratings
with open(args.pre_attentive_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        picture_id = row['picture_id']
        item_id = picture_id.split('_')[0]
        item_name = item_map_id_to_english[item_id]

        if item_name not in item_names:
            item_names.append(item_name)
        
        if item_name not in individual_ratings['pre-attentive']:
            item_name_german = row['item']
            item_map_german_to_english[item_name_german] = item_name
            # if not: add item information to dictionary
            individual_ratings['pre-attentive'][item_name] = []
            individual_ratings['attentive'][item_name] = []
            
        response = response_mapping[row['Response']]
        individual_ratings['pre-attentive'][item_name].append(response)

# read in information from attentive ratings
with open(args.attentive_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:

        for item_name_german, item_name in item_map_german_to_english.items():
            raw_value = row[item_name_german]
            if len(raw_value) > 0:
                # ignore empty entries
                scaled_value = (int(raw_value) / 500) - 1
                individual_ratings['attentive'][item_name].append(scaled_value)

# create aggregated ratings by simply computing the mean
for feature_type, feature_data in individual_ratings.items():
    for item, ratings in feature_data.items():
        aggregated_ratings[feature_type][item] = np.mean(ratings)

# now binarize the scales by taking the top 25% and bottom 25% as positive and negative examples, respectively
classification_output = {}
for feature_type, feature_data in aggregated_ratings.items():
    list_of_tuples = []
    for item, value in feature_data.items():
        list_of_tuples.append((item, value))
    list_of_tuples = sorted(list_of_tuples, key = lambda x: x[1])
    
    negative_cutoff = int(len(list_of_tuples) / 4)
    positive_cutoff = len(list_of_tuples) - negative_cutoff
    positive = list(map(lambda x: x[0], list_of_tuples[positive_cutoff:]))
    negative = list(map(lambda x: x[0], list_of_tuples[:negative_cutoff]))    
    
    classification_output[feature_type] = {'positive': positive, 'negative': negative}
    
    print('Positive examples for {0}'.format(feature_type))
    print(','.join(positive))
    print('Negative examples for {0}'.format(feature_type))
    print(','.join(negative))


# write pickle output
pickle_output = {'individual': individual_ratings, 'aggregated': aggregated_ratings, 'classification': classification_output}
with open(args.output_pickle_file, 'wb') as f_out:
    pickle.dump(pickle_output, f_out)

# output to csv
# ... first: individual ratings
with open(args.output_csv_file_individual, 'w') as f_out:
    f_out.write('item;ratingType;ratings\n')
    for feature_type, feature_data in individual_ratings.items():
        for item, ratings in feature_data.items():
            for rating in ratings:
                f_out.write('{0};{1};{2}\n'.format(item, feature_type, rating))

# ... then: aggregated ratings
with open(args.output_csv_file_aggregated, 'w') as f_out:
    f_out.write('item;ratingType;ratings\n')
    for feature_type, feature_data in aggregated_ratings.items():
        for item, rating in feature_data.items():
            f_out.write('{0};{1};{2}\n'.format(item, feature_type, rating))

# compute intersection of positive and negative scale end
for first_feature_type, second_feature_type in combinations(sorted(classification_output.keys()), 2):

    first_data = classification_output[first_feature_type]
    second_data = classification_output[second_feature_type]
    
    intersection_pos = len(set(first_data['positive']).intersection(second_data['positive']))
    intersection_neg = len(set(first_data['negative']).intersection(second_data['negative']))
    
    print('Comparing classification of {0} to {1}: Intersection Pos {2}, Intersection Neg {3}'.format(first_feature_type, second_feature_type, intersection_pos, intersection_neg))

# creating scatter plots
if args.plot_folder is not None:

    item_ids = [item_map_english_to_id[item] for item in item_names]

    # first read in all the images
    images = None
    if args.image_folder != None:
        images = load_item_images(args.image_folder, item_ids)    
    
    # look at all pairs of feature types
    for first_feature_type, second_feature_type in combinations(sorted(aggregated_ratings.keys()), 2):
        
        # collect the data
        first_vector = []
        second_vector = []
        for item in item_names:
            first_vector.append(aggregated_ratings[first_feature_type][item])
            second_vector.append(aggregated_ratings[second_feature_type][item])
        
        # create scatter plot
        output_file_name = os.path.join(args.plot_folder, 'scatter-{0}-{1}.png'.format(first_feature_type, second_feature_type))        
        create_labeled_scatter_plot(first_vector, second_vector, output_file_name, x_label = first_feature_type, y_label = second_feature_type, images = images, zoom = args.zoom)  
