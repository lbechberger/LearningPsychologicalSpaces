# -*- coding: utf-8 -*-
"""
Aggregates the similarity values and stores it in the form of a matrix in a pickle file.

Created on Tue Dec  4 09:02:18 2018

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from code.util import list_to_string

parser = argparse.ArgumentParser(description='Computing aggregated similarity values')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('output_pickle_file', help = 'path to the output pickle file')
parser.add_argument('output_folder_matrix', help = 'output folder for storing the dissimilarity matrix')
parser.add_argument('output_csv_file', help = 'path to the output CSV file for the aggregated similarities')
parser.add_argument('rating_type', help = 'type of the underlying similarity ratings')
parser.add_argument('-m', '--median', action="store_true", help = 'use median instead of mean')
args = parser.parse_args()

# store aggregator function for later: mean by default, median by request
aggregator = np.median if args.median else np.mean

# load the data set from the pickle file
with open(args.input_file, "rb") as f_in:
    data_set = pickle.load(f_in)

categories = data_set['category_names']
items = []
for category in categories:
    items += data_set['categories'][category]['items']

# compute dissimilarity matrix
dissimilarity_matrix = np.zeros((len(items), len(items)))
similarity_matrix = np.full((len(items), len(items)), np.nan)
category_matrix_raw = []
for i in range(len(categories)):
    category_matrix_raw.append([])
    for j in range(len(categories)):
        category_matrix_raw[i].append([])

number_of_filled_entries = 0
constraints_per_item = {}
for item in items:
    constraints_per_item[item] = 0

# all raw similarity ratings that are used to aggregate the entries of the dissimilarity matrix
all_similarities = []

# dictionary mapping from unique similarity values to their respective counts
pairwise_similarities = {}
def add_pairwise_similarity(similarity_value):
    if similarity_value in pairwise_similarities:
        pairwise_similarities[similarity_value] += 1
    else:
        pairwise_similarities[similarity_value] = 1

for index1, item1 in enumerate(items):
    for index2, item2 in enumerate(items):
        
        if index2 <= index1:
            if index2 == index1:
                number_of_filled_entries += 1
                constraints_per_item[item1] += 1
                similarity_matrix[index1][index2] = 5 # manually set self-similarity to max
            continue
        
        tuple_id = list_to_string([item1, item2])
        if tuple_id not in data_set['similarities'].keys():
            # unknown similarities: simply leave zeroes in there (are treated as "unknown" by MDS algorithm)
            # see https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/manifold/mds.py#L100
            continue
        
        # compute dissimilarity
        similarity_ratings = data_set['similarities'][tuple_id]['values']
        # if we don't have any data: skip this table entry
        if len(similarity_ratings) == 0:
            continue
        
        # aggregate by mean (default) or median (if requested on command line)
        overall_similarity = aggregator(similarity_ratings)
        dissimilarity = 5 - overall_similarity
        
        # add to similarity matrix
        similarity_matrix[index1][index2] = overall_similarity
        similarity_matrix[index2][index1] = overall_similarity        
        
        # add to dissimilarity matrix
        dissimilarity_matrix[index1][index2] = dissimilarity
        dissimilarity_matrix[index2][index1] = dissimilarity
        
        # add to category matrix
        cat_idx1 = categories.index(data_set['items'][item1]['category']) 
        cat_idx2 = categories.index(data_set['items'][item2]['category'])
        category_matrix_raw[cat_idx1][cat_idx2] += similarity_ratings
        if cat_idx1 != cat_idx2:
            category_matrix_raw[cat_idx2][cat_idx1] += similarity_ratings
        
        # keep track of statistics
        number_of_filled_entries += 2
        constraints_per_item[item1] += 1
        constraints_per_item[item2] += 1

        all_similarities += similarity_ratings
        add_pairwise_similarity(overall_similarity)

# aggregate values on category basis
category_matrix = np.zeros((len(categories), len(categories)))
for i in range(len(categories)):
    for j in range(len(categories)):
        overall_similarity = aggregator(category_matrix_raw[i][j])
        category_matrix[i][j] = overall_similarity


# write pickle output
result = {'items': items, 'similarities': similarity_matrix, 'category_similarities': category_matrix,
          'dissimilarities': dissimilarity_matrix, 'categories': categories}

with open(args.output_pickle_file, 'wb') as f_out:
    pickle.dump(result, f_out)

# write csv output for R analysis
aggregator_name = 'median' if args.median else 'mean'
with open(args.output_csv_file, 'w') as f_out:
    f_out.write('pairID,pairType,visualType,ratingType,aggregator,ratings\n')
    for index1, item1 in enumerate(items):
        for index2, item2 in enumerate(items):
            
            if index2 <= index1:
                continue
            tuple_id = list_to_string([item1, item2])
            rating = similarity_matrix[index1,index2]
            pair_type = data_set['similarities'][tuple_id]['relation']
            visual_type = data_set['similarities'][tuple_id]['category_type']
            f_out.write("{0},{1},{2},{3},{4},{5}\n".format(tuple_id, pair_type, visual_type, args.rating_type, aggregator_name, rating))
       

# write dissimilarity matrix output
with open(os.path.join(args.output_folder_matrix, 'distance_matrix.csv'), 'w') as f_out:
    for line in dissimilarity_matrix:
        f_out.write('{0}\n'.format(','.join(map(lambda x: str(x), line))))

with open(os.path.join(args.output_folder_matrix, 'item_names.csv'), 'w') as f_out:
    for item in items:
        f_out.write("{0}\n".format(item))

# analyze matrix
matrix_size = len(items) * len(items)
print("dissimilarity matrix: {0} x {0}, {1} entries, {2} are filled (equals {3}%)".format(len(items), 
          matrix_size, number_of_filled_entries, 100*(number_of_filled_entries / matrix_size)))

average_num_constraints = 0
for item in items:
    num_constraints = constraints_per_item[item]
    print("{0}: {1} constraints".format(item, num_constraints))
    average_num_constraints += num_constraints
print("average number of constraints per item: {0}".format(average_num_constraints / len(items)))

number_of_ties = 0
for value in sorted(pairwise_similarities.keys()):
    count = pairwise_similarities[value]
    number_of_ties += (count * (count - 1)) / 2
print("number of ties (off diagonal, ignoring symmetry) in the matrix: {0} ({1}% of the pairs, {2} distinct values)".format(number_of_ties, 
          100 * (number_of_ties / ((matrix_size * (matrix_size - 1)) / 2)), len(pairwise_similarities.keys())))