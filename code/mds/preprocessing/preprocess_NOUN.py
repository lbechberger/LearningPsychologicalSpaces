# -*- coding: utf-8 -*-
"""
Preprocesses the original distances such that they can be used by the MDS algorithm.

Created on Wed Jan 30 14:15:16 2019

@author: lbechberger
"""

import pickle, argparse, csv
from code.util import list_to_string

parser = argparse.ArgumentParser(description='Preprocessing similarity data of the NOUN study')
parser.add_argument('distance_table', help = 'CSV file containing the distance data of the NOUN study')
parser.add_argument('output_file', help = 'path to the output pickle file')
args = parser.parse_args()

category_info = {}
item_info = {}
similarity_info = {}

cat_name = 'no_category'
category_info[cat_name] = {'visSim' : 'x', 'artificial' : 'art', 'items' : []}

# read dissimilarity matrix from csv file
with open(args.distance_table, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        # first item known by entry in column 'Stimulus'
        item1 = row['Stimulus']
        
        if item1 not in item_info:
            # if not: add item information to dictionary
            item_info[item1] = {'category': cat_name}
        # check whether item is already associated with category
        if item1 not in category_info[cat_name]['items']:
            # if not: do so now
            category_info[cat_name]['items'].append(item1)
        
        # second item known by column header
        for item2 in row.keys():
            if item2 == 'Subject' or item2 == 'Stimulus':
                # not a column header of interest: skip
                continue
            if len(row[item2]) == 0:
                # empty cell: skip
                continue
            
            # nonempty cell
            
            if item2 not in item_info:
                # if not: add item information to dictionary
                item_info[item2] = {'category': cat_name}
            # check whether item is already associated with category
            if item2 not in category_info[cat_name]['items']:
                # if not: do so now
                category_info[cat_name]['items'].append(item2)
            
            # convert distances to similarities for easier processing later on: 
            # reduce [0,1500] to [0,5], then revert order
            distance = float(row[item2])
            similarity = 5 - (distance / 300)            
            
            item_tuple_id = list_to_string([item1, item2])
    
            if item_tuple_id in similarity_info:
                # if we already have similarity information from the previous study: append
                similarity_info[item_tuple_id]['values'].append(similarity)
            else:
                # otherwise: add new line
                similarity_info[item_tuple_id] = {'relation': 'within', 'category_type': 'x', 'values': [similarity]}

for _ , sim_dict in similarity_info.items():
    sim_dict['border'] = len(sim_dict['values'])

output = {'categories': category_info, 'items': item_info, 'similarities': similarity_info, 'category_names': [cat_name]}

# dump everything into a pickle file
with open(args.output_file, "wb") as f_out:
    pickle.dump(output, f_out)