# -*- coding: utf-8 -*-
"""
Read in the ratings about a given psychological feature.

Created on Thu Jan  9 11:48:19 2020

@author: lbechberger
"""

import pickle, argparse, csv
import numpy as np

parser = argparse.ArgumentParser(description='Preprocessing ratings for psychological features')
parser.add_argument('pre_attentive_file', help = 'CSV file containing the pre-attentive feature ratings')
parser.add_argument('attentive_file', help = 'CSV file containing the attentive feature ratings')
parser.add_argument('categories_file', help = 'CSV file containing an ordered list of categories and their desired names')
parser.add_argument('items_file', help = 'CSV file containing a mapping of item names')
parser.add_argument('output_pickle_file', help = 'path to the output pickle file')
parser.add_argument('output_csv_file_individual', help = 'path to the output csv file for individual ratings')
parser.add_argument('output_csv_file_aggregated', help = 'path to the output csv file for aggregated ratings')
parser.add_argument('-o', '--output_folder', help = 'folder where the histograms shall be stored', default = '.')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
args = parser.parse_args()

response_mapping = {'keineAhnung': 0,
                    'länglich': -1, 'gleich': 1, 
                    'gebogen': 1, 'gerade': -1,
                    'vertikal': 1, 'horizontal': -1, 'diagonal1': 0, 'diagonal2': 0}

item_map_id_to_english = {}
item_map_german_to_english = {}
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


# read in information from binary ratings
with open(args.pre_attentive_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        picture_id = row['picture_id']
        item_id = picture_id.split('_')[0]
        item_name = item_map_id_to_english[item_id]
        
        if item_name not in individual_ratings:
            item_name_german = row['item']
            # if not: add item information to dictionary
            individual_ratings['pre-attentive'][item_id] = []
            individual_ratings['attentive'][item_id] = []
            
        response = response_mapping[row['Response']]
        individual_ratings['pre-attentive'][item_id].append(response)

# read in information from continuous ratings
with open(args.attentive_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:

        for item_name_german, item_name in item_map_german_to_english.items():
            value = row[item_name]
            if len(value) > 0:
                # ignore empty entries
                individual_ratings['attentive'][item_name].append(int(value))

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

# TODO: output to csv, creating scatter plots, intersection of pos & neg, adapt shell script, documentation, folder structure

