# -*- coding: utf-8 -*-
"""
Preprocesses the original similarity files such that they can be used by the MDS algorithm.

Call like this: 'python preprocess.py path/to/within.csv path/to/within_between.csv path/to/output.pickle'

Created on Mon Oct 22 13:39:05 2018

@author: lbechberger
"""

import pickle, argparse

parser = argparse.ArgumentParser(description='Preprocessing similarity data')
parser.add_argument('within_file', help = 'CSV file containing data from the within-study (study 1)')
parser.add_argument('within_between_file', help = 'CSV file containing data from the within-between-study (study 2)')
parser.add_argument('output_file', help = 'path to the output pickle file')
args = parser.parse_args()


category_info = {}
item_info = {}
similarity_info = {}

# first only read within category information
print("Reading {0}...".format(args.within_file))

with open(args.within_file, 'r') as f:
    for line in f:
        # ignore header
        if line.startswith("Combi_category"):
            continue
        
        tokens = line.replace('\n','').split(',')
        
        # check whether the category is already known
        category = tokens[0]
        if category not in category_info:
            # if not: add category information to dictionary
            category_info[category] = {'visSim': tokens[1], 'artificial': tokens[2], 'items':[]}
        
        # triples: id, name, category
        item1 = (tokens[3], tokens[4], category)
        item2 = (tokens[5], tokens[6], category)

        for item in [item1, item2]:
            # check whether the items are already known
            if item[0] not in item_info:
                # if not: add item information to dictionary
                item_info[item[0]] = {'name': item[1], 'category': item[2]}
            # check whether item is already associated with category
            if item[0] not in category_info[category]['items']:
                # if not: do so now
                category_info[category]['items'].append(item[0])
        
        # get a list of all the similarity values (remove empty entries, then convert to int) and store them
        similarity_values = list(map(lambda x: int(x), filter(None, tokens[7:])))
        similarity_info[str(sorted([tokens[3], tokens[5]]))] = {'relation': 'within', 'values': similarity_values, 'border':len(similarity_values)}

# now read within_between category information
print("Reading {0}...".format(args.within_between_file))
with open(args.within_between_file, 'r') as f:
    for line in f:
        # ignore header
        if line.startswith("Relation"):
            continue
        
        tokens = line.replace('\n', '').split(',')
        
        # tuples: id, name
        item1 = (tokens[4], tokens[5])
        item2 = (tokens[8], tokens[9])

        item_tuple_id = str(sorted([tokens[4], tokens[8]]))
        
        for item in [item1, item2]:
            # check whether the items are already known (they should be by now!)
            if item[0] not in item_info:
                raise Exception("unknown item!")
        
        # get a list of all the similarity values (remove empty entries, then convert to int) and store them
        similarity_values = list(map(lambda x: int(x), filter(None, tokens[12:])))
        
        if item_tuple_id in similarity_info:
            # if we already have similarity information from the previous study: append
            similarity_info[item_tuple_id]['values'] += similarity_values
        else:
            # otherwise: add new line
            similarity_info[item_tuple_id] = {'relation': tokens[0], 'values': similarity_values, 'border':0}

# now write everything into a pickle file
print("Writing output...")
output = {'categories': category_info, 'items': item_info, 'similarities': similarity_info}

# dump everything into a pickle file
with open(args.output_file, "wb") as f:
    pickle.dump(output, f)