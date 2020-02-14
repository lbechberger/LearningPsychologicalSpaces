# -*- coding: utf-8 -*-
"""
Computes the similarity values for the given subset of data and stores it in the form of a matrix in a pickle file.

Created on Tue Dec  4 09:02:18 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from code.util import select_data_subset, find_limit

parser = argparse.ArgumentParser(description='Computing aggregated similarity values')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('output_file', help = 'path to the output pickle file')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
parser.add_argument('-m', '--median', action="store_true", help = 'use median instead of mean')
parser.add_argument('-l', '--limit', action="store_true", help = 'limit the number of similarity ratings to take into account')
parser.add_argument('-v', '--limit_value', type = int, default = 0, help = 'limit value to use')
parser.add_argument('-p', '--plot', action="store_true", help = 'plot two histograms of the distance values')
args = parser.parse_args()

np.random.seed(42) # fixed random seed to ensure reproducibility

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

# select subset of overall data set
items_of_interest, item_names, _ =  select_data_subset(args.subset, data_set) 

# set limit (if necessary)
if args.limit:
    if args.limit_value == 0:
        limit = find_limit(args.subset, data_set, items_of_interest)
        print("Using a computed limit of {0}".format(limit))
    else:
        limit = args.limit_value
        print("Using a given limit of {0}".format(limit))

# compute dissimilarity matrix
dissimilarity_matrix = np.zeros((len(items_of_interest), len(items_of_interest)))
similarity_matrix = np.full((len(items_of_interest), len(items_of_interest)), np.nan)
number_of_filled_entries = 0
constraints_per_item = {}
for item in items_of_interest:
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

for index1, item1 in enumerate(items_of_interest):
    for index2, item2 in enumerate(items_of_interest):
        
        if index2 <= index1:
            if index2 == index1:
                number_of_filled_entries += 1
                constraints_per_item[item1] += 1
                similarity_matrix[index1][index2] = 5 # manually set self-similarity to max
            continue
        
        tuple_id = str(sorted([item1, item2]))
        if tuple_id not in data_set['similarities'].keys():
            # unknown similarities: simply leave zeroes in there (are treated as "unknown" by MDS algorithm)
            # see https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/manifold/mds.py#L100
            continue
        
        # compute dissimilarity
        similarity_ratings = data_set['similarities'][tuple_id]['values']
        if args.subset == "between":
            # remove everything from first study
            border = data_set['similarities'][tuple_id]['border']
            similarity_ratings = similarity_ratings[border:]
        elif args.subset == "within":
            # remove everything from second study
            border = data_set['similarities'][tuple_id]['border']
            similarity_ratings = similarity_ratings[:border]

        # only take a random subset of the ratings (up to the limit)
        if args.limit:
            np.random.shuffle(similarity_ratings)
            similarity_ratings = similarity_ratings[:limit]

        # if we don't have any data: skip this table entry
        if len(similarity_ratings) == 0:
            continue
        
        # aggregate by mean (default) or median (if requested on command line)
        if args.median:
            overall_similarity = np.median(similarity_ratings)
        else:
            overall_similarity = np.mean(similarity_ratings)
        dissimilarity = 5 - overall_similarity
        
        # add to similarity matrix
        similarity_matrix[index1][index2] = overall_similarity
        similarity_matrix[index2][index1] = overall_similarity        
        
        # add to dissimilarity matrix
        dissimilarity_matrix[index1][index2] = dissimilarity
        dissimilarity_matrix[index2][index1] = dissimilarity
        
        # keep track of statistics
        number_of_filled_entries += 2
        constraints_per_item[item1] += 1
        constraints_per_item[item2] += 1

        all_similarities += similarity_ratings
        add_pairwise_similarity(overall_similarity)

# analyze matrix
matrix_size = len(items_of_interest) * len(items_of_interest)
print("dissimilarity matrix: {0} x {0}, {1} entries, {2} are filled (equals {3}%)".format(len(items_of_interest), 
          matrix_size, number_of_filled_entries, 100*(number_of_filled_entries / matrix_size)))

average_num_constraints = 0
for item, num_constraints in constraints_per_item.items():
    print("{0}: {1} constraints".format(item, num_constraints))
    average_num_constraints += num_constraints
print("average number of constraints per item: {0}".format(average_num_constraints / len(items_of_interest)))

number_of_ties = 0
for value, count in pairwise_similarities.items():
    number_of_ties += (count * (count - 1)) / 2
print("number of ties (off diagonal, ignoring symmetry) in the matrix: {0} ({1}% of the pairs, {2} distinct values)".format(number_of_ties, 
          100 * (number_of_ties / ((matrix_size * (matrix_size - 1)) / 2)), len(pairwise_similarities.keys())))

result = {'items': items_of_interest, 'item_names': item_names, 'similarities': similarity_matrix, 'dissimilarities': dissimilarity_matrix}

with open(args.output_file, 'wb') as f:
    pickle.dump(result, f)

# plot the distribution of distances in the distance matrix
if args.plot:
    from matplotlib import pyplot as plt

    output_path = args.output_file.split('.')[0]    
    
    bin_sizes = [21, 5]
    
    for bin_size in bin_sizes:
        plt.hist(all_similarities, bins=bin_size)
        plt.title('Distribution of Raw Similarity Ratings')
        plt.xlabel('Similarity')
        plt.ylabel('Number of Occurences')
        plt.savefig('{0}-distr-{1}.png'.format(output_path, bin_size), bbox_inches='tight', dpi=200)
        plt.close()
    
        dissimilarity_values = dissimilarity_matrix.reshape((-1,1))
        plt.hist(dissimilarity_values, bins=bin_size)
        plt.title('Distribution of Values in Global Dissimilarity Matrix')
        plt.xlabel('Dissimilarity')
        plt.ylabel('Number of Occurences')
        plt.savefig('{0}-matrix-{1}.png'.format(output_path, bin_size), bbox_inches='tight', dpi=200)
        plt.close()          