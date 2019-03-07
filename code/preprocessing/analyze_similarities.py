# -*- coding: utf-8 -*-
"""
Analyze the distribution of similarity ratings in the data set

Created on Mon Jan 14 14:00:32 2019

@author: lbechberger
"""
from matplotlib import pyplot as plt

import pickle, argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analyzing similarity data')
parser.add_argument('input_file', help = 'pickle file containing the preprocessed data')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default = "all")
parser.add_argument('-o', '--output_path', help = 'path where to store the figures', default = './')
parser.add_argument('-m', '--median', action = 'store_true', help = 'use median instead of mean for matrix aggregation')
args = parser.parse_args()

np.random.seed(42) # fixed random seed to ensure reproducibility

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

item_ids = list(data_set['items'].keys())
category_names = list(data_set['categories'].keys())

if args.subset == "all":
    # use all the similarity ratings that we have    
    
    items_of_interest = list(item_ids)
    categories_of_interest = list(category_names)

elif args.subset == "between":
    # only use the similarity ratings from the 'between' file

    items_of_interest = []   
    
    for idx1, item1 in enumerate(item_ids):
        for idx2, item2 in enumerate(item_ids):
            
            if idx2 <= idx1:
                continue
            
            tuple_id = str(sorted([item1, item2]))
            if tuple_id in data_set['similarities']:
                border = data_set['similarities'][tuple_id]['border']
                between_ratings = data_set['similarities'][tuple_id]['values'][border:]
                if len(between_ratings) > 0:
                    items_of_interest.append(item1)
                    items_of_interest.append(item2)
    
    items_of_interest = list(set(items_of_interest)) # remove duplicates
    categories_of_interest = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))

elif args.subset == "within":
    # only use the similarity ratings from the 'within' file
    items_of_interest = []   
    
    for idx1, item1 in enumerate(item_ids):
        for idx2, item2 in enumerate(item_ids):
            
            if idx2 <= idx1:
                continue
            
            tuple_id = str(sorted([item1, item2]))
            if tuple_id in data_set['similarities']:
                border = data_set['similarities'][tuple_id]['border']
                between_ratings = data_set['similarities'][tuple_id]['values'][:border]
                if len(between_ratings) > 0:
                    items_of_interest.append(item1)
                    items_of_interest.append(item2)
    
    items_of_interest = list(set(items_of_interest)) # remove duplicates
    categories_of_interest = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))
    
elif args.subset == "cats":
    # consider only the categories from the second study, but use all items within them
    categories_of_interest = ["C03_Elektrogeräte", "C04_Gebäude", "C05_Gemüse", "C06_Geschirr", "C07_Insekten", 
                                   "C10_Landtiere", "C12_Oberkörperbekleidung", "C13_Obst", "C14_Pflanzen", 
                                   "C19_Straßenfahrzeuge", "C21_Vögel", "C25_Werkzeug"]
    items_of_interest = []
    for item in item_ids:
        if data_set['items'][item]['category'] in categories_of_interest:
            items_of_interest.append(item)

categories_of_interest = sorted(categories_of_interest)

# collect overall statistics
similarity_ranges = []
similarity_stds = []

for idx1, item1 in enumerate(items_of_interest):
    for idx2, item2 in enumerate(items_of_interest):
        
        if idx2 <= idx1:
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

            if len(similarity_ratings) == 0:
                continue

            # analyze range and standard deviation of the ratings        
            similarity_range = max(similarity_ratings) - min(similarity_ratings)
            similarity_std = np.std(similarity_ratings)
            #print('{0} range: {1} std: {2}'.format(tuple_id, similarity_range, similarity_std))
            similarity_ranges.append(similarity_range)
            similarity_stds.append(similarity_std)
        

# plot histograms of ranges and standard deviations
plt.hist(similarity_ranges, bins=21)
plt.title('distribution of similarity ranges')
plt.savefig(args.output_path + 'range.png', bbox_inches='tight', dpi=200)
plt.close()

plt.hist(similarity_stds, bins=21)
plt.title('distribution of similarity std')
plt.savefig(args.output_path + 'std.png', bbox_inches='tight', dpi=200)
plt.close()

similarity_matrix = []
for i in range(len(categories_of_interest)):
    similarity_matrix.append([])
    for j in range(len(categories_of_interest)):
        similarity_matrix[i].append([])

# collect category-level statistics
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

all_within = []
all_between = []

for i in range(len(categories_of_interest)):
    for j in range(len(categories_of_interest)):
        if i == j:
            all_within += similarity_matrix[i][j]
        elif i < j:
            all_between += similarity_matrix[i][j]

# plot histograms of within vs between categories
plt.hist(all_within, bins=21)
plt.title('distribution of similarity ranges within categories')
plt.savefig(args.output_path + 'within.png', bbox_inches='tight', dpi=200)
plt.close()

plt.hist(all_between, bins=21)
plt.title('distribution of similarity ranges between categories')
plt.savefig(args.output_path + 'between.png', bbox_inches='tight', dpi=200)
plt.close()
