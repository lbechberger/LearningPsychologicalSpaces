# -*- coding: utf-8 -*-
"""
Reads in all pickle files about features and exports compact csv files.

Created on Wed Aug 19 11:56:40 2020

@author: lbechberger
"""

import pickle, argparse, os
from code.util import list_to_string

parser = argparse.ArgumentParser(description='Exporting feature ratings in compact csv files')
parser.add_argument('input_folder_features', help = 'input folder with the pickle files containing feature information')
parser.add_argument('input_pickle_file_similarities', help = 'input pickle file with the individual similarity ratings')
parser.add_argument('sim_rating_type', help = 'type of the underlying similarity ratings')
parser.add_argument('output_csv_file_individual', help = 'path to the output csv file for individual ratings')
parser.add_argument('output_csv_file_aggregated', help = 'path to the output csv file for aggregated ratings')
parser.add_argument('output_csv_file_combined', help = 'path to the output csv file for the combination of similarity and feature information')
args = parser.parse_args()

data = {}
feature_names = []
rating_types = ['pre-attentive', 'attentive']

# read in all the '.pickle' files in the feature input folder
for file_name in os.listdir(args.input_folder_features):
    if file_name.endswith('.pickle'):
        feature_name = file_name.split('.')[0]
        with open(os.path.join(args.input_folder_features, file_name), 'rb') as f_in:
            feature_data = pickle.load(f_in)

        data[feature_name] = feature_data
        feature_names.append(feature_name)

# read in the aggregated similarity ratings
with open(args.input_pickle_file_similarities, 'rb') as f_in:
    similarity_data = pickle.load(f_in)

# first write the individual ratings
with open(args.output_csv_file_individual, 'w') as f_out:
    f_out.write('item,ratingType,feature,ratings\n')
    for feature_name in feature_names:
        for rating_type in rating_types:
            item_dict = data[feature_name]['individual'][rating_type]
            items = sorted(item_dict.keys())
            for item in items:
                for rating in item_dict[item]:
                    f_out.write('{0},{1},{2},{3}\n'.format(item, rating_type, feature_name, rating))

# then write the aggregated ratings
line_info = {}
for feature_name in feature_names:
    for rating_type in rating_types:
        item_dict = data[feature_name]['aggregated'][rating_type]
        items = sorted(item_dict.keys())
        for item in items:
            if item not in line_info:
                line_info[item] = {}
            if rating_type not in line_info[item]:
                line_info[item][rating_type] = {}
            line_info[item][rating_type][feature_name] = item_dict[item]

with open(args.output_csv_file_aggregated, 'w') as f_out:
    f_out.write('item,ratingType,{0}\n'.format(','.join(feature_names)))
    for item in items:
        for rating_type in rating_types:
            rating_dict = line_info[item][rating_type]
            feature_ratings = [rating_dict[feature] for feature in feature_names]
            f_out.write('{0},{1},{2}\n'.format(item, rating_type, ",".join(map(lambda x: str(x), feature_ratings))))


# finally write the combined ratings
with open(args.output_csv_file_combined, 'w') as f_out:
    
    # write header
    f_out.write('pairID,pairType,visualType,ratingType,ratings')
    for i in [1,2]:
        for feature in feature_names:
            for rating_type in rating_types:
                f_out.write(',item{0}_{1}_{2}'.format(i, feature, rating_type))
    f_out.write('\n')
    
    # construct list of all items
    items = [item for cat in similarity_data['category_names'] for item in similarity_data['categories'][cat]['items']]
    
    for index1, item1 in enumerate(items):
        for index2, item2 in enumerate(items):
            
            if index2 <= index1:
                continue
            tuple_id = list_to_string([item1, item2])
            pair_type = similarity_data['similarities'][tuple_id]['relation']
            visual_type = similarity_data['similarities'][tuple_id]['category_type']
            
            feature_values = []
            for item in [item1, item2]:
                for feature in feature_names:
                    for rating_type in rating_types:
                        feature_values.append(line_info[item][rating_type][feature])
            
            for sim_rating in similarity_data['similarities'][tuple_id]['values']:
                f_out.write("{0},{1},{2},{3},{4},{5}\n".format(tuple_id, pair_type, visual_type, args.sim_rating_type, sim_rating, ",".join(map(lambda x: str(x), feature_values))))

