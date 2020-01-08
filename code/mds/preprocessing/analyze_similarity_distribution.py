# -*- coding: utf-8 -*-
"""
Analyze the distribution of similarity ratings in the data set

Created on Mon Jan 14 14:00:32 2019

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from scipy.stats import mannwhitneyu
from code.util import select_data_subset, find_limit

parser = argparse.ArgumentParser(description='Analyzing similarity data')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default = "all")
parser.add_argument('-l', '--limit', action = 'store_true', help = 'limit the number of similarity ratings to take into account')
parser.add_argument('-v', '--limit_value', type = int, default = 0, help = 'limit value to use')
parser.add_argument('-m', '--median', action = 'store_true', help = 'use median instead of mean for matrix aggregation')
args = parser.parse_args()

np.random.seed(42) # fixed random seed to ensure reproducibility

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

# select subset of overall data set
items_of_interest, item_names, categories_of_interest = select_data_subset(args.subset, data_set)

# set limit (if necessary)
if args.limit:
    if args.limit_value == 0:
        limit = find_limit(args.subset, data_set, items_of_interest)
        print("Using a computed limit of {0}".format(limit))
    else:
        limit = args.limit_value
        print("Using a given limit of {0}".format(limit))

# collect category-level statistics
similarity_matrix = []
for i in range(len(categories_of_interest)):
    similarity_matrix.append([])
    for j in range(len(categories_of_interest)):
        similarity_matrix[i].append([])

for cat_idx1, cat1 in enumerate(categories_of_interest):
    for cat_idx2, cat2 in enumerate(categories_of_interest):
       
        if cat_idx2 < cat_idx1:
            continue
        
        within_sim_1 = []
        within_sim_2 = []
        between_sim = []
        
        for itm_idx1, item1 in enumerate(data_set['categories'][cat1]['items']):
            for itm_idx2, item2 in enumerate(data_set['categories'][cat1]['items']):

                if itm_idx2 <= itm_idx1:
                    continue
                
                tuple_id = str(sorted([item1, item2]))   
                if tuple_id in data_set['similarities']:
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
       
                    within_sim_1 += similarity_ratings
                    
        for itm_idx1, item1 in enumerate(data_set['categories'][cat2]['items']):
            for itm_idx2, item2 in enumerate(data_set['categories'][cat2]['items']):

                if itm_idx2 <= itm_idx1:
                    continue
                
                tuple_id = str(sorted([item1, item2]))   
                if tuple_id in data_set['similarities']:
                    similarity_ratings = data_set['similarities'][tuple_id]['values']
                    
                    if args.subset == "between":
                        # remove everything from first study
                        border = data_set['similarities'][tuple_id]['border']
                        similarity_ratings = similarity_ratings[border:]
                    elif args.subset == "within":
                        # remove everything from second study
                        border = data_set['similarities'][tuple_id]['border']
                        similarity_ratings = similarity_ratings[:border]
        
                    if args.limit:
                        np.random.shuffle(similarity_ratings)
                        similarity_ratings = similarity_ratings[:limit]

                    within_sim_2 += similarity_ratings
                
        for itm_idx1, item1 in enumerate(data_set['categories'][cat1]['items']):
            for itm_idx2, item2 in enumerate(data_set['categories'][cat2]['items']):

                if itm_idx2 <= itm_idx1:
                    continue
                
                tuple_id = str(sorted([item1, item2]))   
                if tuple_id in data_set['similarities']:
                    similarity_ratings = data_set['similarities'][tuple_id]['values']
                    
                    if args.subset == "between":
                        # remove everything from first study
                        border = data_set['similarities'][tuple_id]['border']
                        similarity_ratings = similarity_ratings[border:]
                    elif args.subset == "within":
                        # remove everything from second study
                        border = data_set['similarities'][tuple_id]['border']
                        similarity_ratings = similarity_ratings[:border]
        
                    if args.limit:
                        np.random.shuffle(similarity_ratings)
                        similarity_ratings = similarity_ratings[:limit]

                    between_sim += similarity_ratings
        
        similarity_matrix[cat_idx1][cat_idx1] = within_sim_1
        similarity_matrix[cat_idx2][cat_idx2] = within_sim_2
        similarity_matrix[cat_idx1][cat_idx2] = between_sim
        similarity_matrix[cat_idx2][cat_idx1] = between_sim

# print out average similarity ratings on category-level
print(',' + ','.join(map(lambda x: '{0}({1})'.format(x, data_set['categories'][x]['visSim']), categories_of_interest)))
for i in range(len(categories_of_interest)):
    mean_list = []
    for j in range(len(categories_of_interest)):
        if args.median:
            mean_list.append(np.median(similarity_matrix[i][j]))
        else:
            mean_list.append(np.mean(similarity_matrix[i][j]))
    print("{0}({1})".format(categories_of_interest[i], data_set['categories'][categories_of_interest[i]]['visSim']) + ',' + ','.join(map(lambda x: str(x), mean_list)))


list_of_within_similarities = [] # diagonal entries of the matrix, annotated with category and their visSim rating
sim_ratings = [] # all within-category similarity ratings for the 'Sim' categories
dis_ratings = [] # all within-category similarity ratings for the 'Dis' categories

for i in range(len(categories_of_interest)):
    category = categories_of_interest[i]
    vis_sim = data_set['categories'][category]['visSim']
    similarities = similarity_matrix[i][i]
    
    # fill buckets of similarity ratings
    if vis_sim == 'Sim':
        sim_ratings += similarities
    elif vis_sim == 'Dis':
        dis_ratings += similarities
    
    # fill list of diagonal entries
    if args.median:
        aggregated_value = np.median(similarities)
    else:
        aggregated_value = np.mean(similarities)
    list_of_within_similarities.append((aggregated_value, category, vis_sim))

# statistical test comparing 'Sim' to 'Dis' categories
if args.median:
    overall_sim = np.median(sim_ratings)
    overall_dis = np.median(dis_ratings)
else:
    overall_sim = np.mean(sim_ratings)
    overall_dis = np.mean(dis_ratings)
# use MannWhitney-U as it only assumes an ordinal scale
mannwhitneyu_stat, mannwhitneyu_p = mannwhitneyu(sim_ratings, dis_ratings)
print('\nStatistical analysis of differences: p = {0} (stat: {1}) - sim {2} vs dis {3}'.format(mannwhitneyu_p, mannwhitneyu_stat, overall_sim, overall_dis))


# print out sorted list of within-category similarities
list_of_within_similarities = sorted(list_of_within_similarities, key = lambda x: x[0])
print('\nCategories sorted by within-similarity:')
print('similarity,category,visSim')
for entry in list_of_within_similarities:
    print('{0},{1},{2}'.format(entry[0], entry[1], entry[2]))