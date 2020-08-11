# -*- coding: utf-8 -*-
"""
Helper script for exporting the relevant similarity information from the data 
set to a csv file suitable for statistical analysis with R scripts.

Created on Tue Aug 11 08:32:43 2020

@author: lbechberger
"""
         
import pickle, argparse

parser = argparse.ArgumentParser(description='Creating CSV file for R code')
parser.add_argument('input_file', help = 'the input file to use')
parser.add_argument('output_file', help = 'the output file to use')
parser.add_argument('rating_type', help = 'name of the rating study')
args = parser.parse_args()

with open(args.input_file, 'rb') as f:
    input_data = pickle.load(f)

similarities = input_data['similarities']

with open(args.output_file, 'w') as f:
    f.write("pairID;pairType;visualType;ratingType;ratings\n")
    for key, value in similarities.items():
        pairID = key
        pairType = value['relation']
        ratings = value['values'][value['border']:]
        visualType = value['category_type']
        
        for rating in ratings:
            f.write("{0};{1};{2};{3};{4}\n".format(pairID, pairType, visualType, args.rating_type, rating))
