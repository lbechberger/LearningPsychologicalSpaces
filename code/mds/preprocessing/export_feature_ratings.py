# -*- coding: utf-8 -*-
"""
Reads in all pickle files about features and exports compact csv files.

Created on Wed Aug 19 11:56:40 2020

@author: lbechberger
"""

import pickle, argparse, os

parser = argparse.ArgumentParser(description='Exporting feature ratings in compact csv files')
parser.add_argument('input_folder', help = 'input folder with the pickle files containing feature information')
parser.add_argument('output_csv_file_individual', help = 'path to the output csv file for individual ratings')
parser.add_argument('output_csv_file_aggregated', help = 'path to the output csv file for aggregated ratings')
args = parser.parse_args()

data = {}

# read in all the '.pickle' files in the input folder
for file_name in os.listdir(args.input_folder):
    if file_name.endswith('.pickle'):
        feature_name = file_name.split('.')[0]
        with open(os.path.join(args.input_folder, file_name), 'rb') as f_in:
            feature_data = pickle.load(f_in)

        data[feature_name] = feature_data

# first write the individual ratings
with open(args.output_csv_file_individual, 'w') as f_out:
    f_out.write('item;ratingType;feature;ratings\n')
    for feature_name, feature_data in data.items():
        for rating_type, inner_dict in feature_data['individual'].items():
            for item, ratings in inner_dict.items():
                for rating in ratings:
                    f_out.write('{0};{1};{2};{3}\n'.format(item, rating_type, feature_name, rating))

# then write the aggregated ratings
line_info = {}
for feature_name, feature_data in data.items():
    for rating_type, inner_dict in feature_data['aggregated'].items():
        for item, rating in inner_dict.items():
            if item not in line_info:
                line_info[item] = {}
            if rating_type not in line_info[item]:
                line_info[item][rating_type] = {}
            line_info[item][rating_type][feature_name] = rating

feature_names = sorted(data.keys())
with open(args.output_csv_file_aggregated, 'w') as f_out:
    f_out.write('item;ratingType;{0}\n'.format(';'.join(feature_names)))
    for item, item_dict in line_info.items():
        for rating_type, rating_dict in item_dict.items():
            feature_ratings = [rating_dict[feature] for feature in feature_names]
            f_out.write('{0};{1};{2}\n'.format(item, rating_type, ";".join(map(lambda x: str(x), feature_ratings))))