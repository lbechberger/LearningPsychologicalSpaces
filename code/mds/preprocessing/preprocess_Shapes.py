# -*- coding: utf-8 -*-
"""
Preprocesses the original similarity files such that they can be used by the MDS algorithm.

Call like this: 'python preprocess.py path/to/within.csv path/to/within_between.csv path/to/output.pickle'

Created on Mon Oct 22 13:39:05 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from code.util import select_data_subset, find_limit

parser = argparse.ArgumentParser(description='Preprocessing similarity data of the Shapes study')
parser.add_argument('within_file', help = 'CSV file containing data from the within-study (study 1)')
parser.add_argument('within_between_file', help = 'CSV file containing data from the within-between-study (study 2)')
parser.add_argument('categories_file', help = 'CSV file containing an ordered list of categories and their desired names')
parser.add_argument('items_file', help = 'CSV file containing a mapping of item names')
parser.add_argument('output_pickle_file', help = 'path to the output pickle file')
parser.add_argument('output_csv_file', help = 'path to the output csv file')
parser.add_argument('rating_type', help = 'type of the underlying similarity ratings')
parser.add_argument('-r', '--reverse', action = 'store_true', help = 'use distances instead of similarities')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
parser.add_argument('-l', '--limit', action="store_true", help = 'limit the number of similarity ratings to take into account')
parser.add_argument('-v', '--limit_value', type = int, default = 0, help = 'limit value to use')
parser.add_argument('--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

category_names = []
category_map = {}
item_map = {}
vis_sim_map = {'Sim': 'VC', 'Dis': 'VV', 'x': 'x'}

category_info = {}
item_info = {}
similarity_info = {}

# set seed if needed
if args.seed is not None:
   np.random.seed(args.seed)


# read in the category names
with open(args.categories_file, 'r') as f_in:
    for line in f_in:
        tokens = line.replace('\n','').split(',')
        
        category_map[tokens[0]] = tokens[1]
        category_names.append(tokens[1])

# read in the item names
with open(args.items_file, 'r') as f_in:
    for line in f_in:
        tokens = line.replace('\n','').split(',')
        item_map[tokens[0]] = tokens[1]
        
# first only read within category information
with open(args.within_file, 'r') as f_in:
    for line in f_in:
        # ignore header
        if line.startswith("Combi_category"):
            continue
        
        tokens = line.replace('\n','').split(',')
        
        # skip categories that have not been specified in the categories file
        if tokens[0] not in category_map:
            continue
        category = category_map[tokens[0]]
        
        # check whether the category is already known
        if category not in category_info:
            # if not: add category information to dictionary
            category_info[category] = {'visSim': vis_sim_map[tokens[1]], 'artificial': tokens[2], 'items':[]}
        
        
        # tuples: name, category
        item1 = (item_map[tokens[3]], category)
        item2 = (item_map[tokens[5]], category)

        for item in [item1, item2]:
            # check whether the items are already known
            if item[0] not in item_info:
                # if not: add item information to dictionary
                item_info[item[0]] = {'category': item[1]}
            # check whether item is already associated with category
            if item[0] not in category_info[category]['items']:
                # if not: do so now
                category_info[category]['items'].append(item[0])
                category_info[category]['items'] = sorted(category_info[category]['items'])
        
        # get a list of all the similarity values (remove empty entries, then convert to int) and store them
        similarity_values = list(map(lambda x: int(x), filter(None, tokens[7:])))
        if args.reverse:
            similarity_values = list(map(lambda x: 6 - x, similarity_values))
        similarity_info[str(sorted([item_map[tokens[3]], item_map[tokens[5]]]))] = {'relation': 'within', 'category_type': vis_sim_map[tokens[1]], 'values': similarity_values, 'border':len(similarity_values)}

# now read within_between category information
with open(args.within_between_file, 'r') as f_in:
    for line in f_in:
        # ignore header
        if line.startswith("Relation"):
            continue
        
        tokens = line.replace('\n', '').split(',')
        
        # convert into readable name
        item1 = item_map[tokens[4]]
        item2 = item_map[tokens[8]]

        item_tuple_id = str(sorted([item1, item2]))
        
        for item in [item1, item2]:
            # check whether the items are already known (they should be by now!)
            if item not in item_info:
                raise Exception("unknown item!")
        
        # get a list of all the similarity values (remove empty entries, then convert to int) and store them
        similarity_values = list(map(lambda x: int(x), filter(None, tokens[12:])))
        if args.reverse:
            similarity_values = list(map(lambda x: 6 - x, similarity_values))
        
        # transform information about category type
        category_type = 'Mix'
        if tokens[0] == 'within':
            if tokens[1] == 'visDis':
                category_type = 'VV'
            else:
                category_type = 'VC'
        
        if item_tuple_id in similarity_info:
            # if we already have similarity information from the previous study: append
            similarity_info[item_tuple_id]['values'] += similarity_values
        else:
            # otherwise: add new line
            similarity_info[item_tuple_id] = {'relation': tokens[0], 'category_type': category_type, 'values': similarity_values, 'border':0}

# summarize everything in one big dictionary
raw_data = {'categories': category_info, 'items': item_info, 'similarities': similarity_info, 'category_names': category_names}

# select subset of overall data set
items_of_interest, categories_of_interest =  select_data_subset(args.subset, raw_data) 

# set limit (if necessary)
if args.limit:
    if args.limit_value == 0:
        limit = find_limit(args.subset, raw_data, items_of_interest)
    else:
        limit = args.limit_value

# filter data accordingly
filtered_category_info = {}
filtered_item_info = {}
filtered_similarity_info = {}

# ... first the items
for item in items_of_interest:
    filtered_item_info[item] = item_info[item]

# ... then the categories
for category in categories_of_interest:
    ci = category_info[category]
    items_in_cat = [item for item in items_of_interest if item in ci['items']]
    ci['items'] = sorted(items_in_cat)
    filtered_category_info[category] = ci

# ... and finally the ratings
for index1, item1 in enumerate(items_of_interest):
    for index2, item2 in enumerate(items_of_interest):
        
        if index2 <= index1:
            continue
        tuple_id = str(sorted([item1, item2]))
        
        si = similarity_info[tuple_id]
        ratings = si['values']

        if args.subset == "between":
            # remove everything from first study
            ratings = ratings[si['border']:]
        elif args.subset == "within":
            # remove everything from second study
            ratings = ratings[:si['border']]

        # only take a random subset of the ratings (up to the limit)
        if args.limit:
            np.random.shuffle(ratings)
            ratings = ratings[:limit]
        
        si['values'] = ratings
        del si['border']
        filtered_similarity_info[tuple_id] = si
        

# now write everything into a pickle file
output = {'categories': filtered_category_info, 'items': filtered_item_info, 'similarities': filtered_similarity_info, 'category_names': categories_of_interest}
with open(args.output_pickle_file, "wb") as f_out:
    pickle.dump(output, f_out)
    
# ... and also into a csv file
with open(args.output_csv_file, 'w') as f_out:
    f_out.write('pairID;pairType;visualType;ratingType;ratings\n')
    for index1, item1 in enumerate(items_of_interest):
        for index2, item2 in enumerate(items_of_interest):
            
            if index2 <= index1:
                continue
            tuple_id = str(sorted([item1, item2]))
            pair_type = filtered_similarity_info[tuple_id]['relation']
            visual_type = filtered_similarity_info[tuple_id]['category_type']
            
            for rating in filtered_similarity_info[tuple_id]['values']:
                f_out.write("{0};{1};{2};{3};{4}\n".format(tuple_id, pair_type, visual_type, args.rating_type, rating))

