# -*- coding: utf-8 -*-
"""
Runs MDS on the given data set with the given configuration.

Created on Mon Oct 22 14:44:04 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import os

parser = argparse.ArgumentParser(description='MDS for shapes')
parser.add_argument('input_file', help = 'the input file to use')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
parser.add_argument('-n', '--n_init', type = int, help = 'number random starts', default = 4)
parser.add_argument('-d', '--dims', type = int, help = 'highest number of dimensions to check', default = 20)
parser.add_argument('-i', '--max_iter', type = int, help = 'maximum number of iterations', default = 300)
parser.add_argument('-e', '--export', help = 'path for export', default = None)
args = parser.parse_args()


# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

item_ids = list(data_set['items'].keys())
    
if args.subset == "all":
    # use all the similarity ratings that we have    
    
    items_of_interest = list(item_ids)

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
    
elif args.subset == "cats":
    # consider only the categories from the second study, but use all items within them
    second_study_categories = ["C03_Elektrogeräte", "C04_Gebäude", "C05_Gemüse", "C06_Geschirr", "C07_Insekten", 
                                   "C10_Landtiere", "C12_Oberkörperbekleidung", "C13_Obst", "C14_Pflanzen", 
                                   "C19_Straßenfahrzeuge", "C21_Vögel", "C25_Werkzeug"]
    items_of_interest = []
    for item in item_ids:
        if data_set['items'][item]['category'] in second_study_categories:
            items_of_interest.append(item)

# no matter which subset was used: create list of item names
item_names = list(map(lambda x: data_set['items'][x]['name'], items_of_interest))


# compute dissimilarity matrix
dissimilarity_matrix = np.zeros((len(items_of_interest), len(items_of_interest)))
number_of_filled_entries = 0
constraints_per_item = {}
for item in items_of_interest:
    constraints_per_item[item] = 0

for index1, item1 in enumerate(items_of_interest):
    for index2, item2 in enumerate(items_of_interest):
        
        if index2 <= index1:
            if index2 == index1:
                number_of_filled_entries += 1
                constraints_per_item[item1] += 1
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
        
        if len(similarity_ratings) == 0:
            print(item1, item2)
        overall_similarity = np.mean(similarity_ratings)
        dissimilarity = 5 - overall_similarity
        
        # add to matrix
        dissimilarity_matrix[index1][index2] = dissimilarity
        dissimilarity_matrix[index2][index1] = dissimilarity
        number_of_filled_entries += 2
        constraints_per_item[item1] += 1
        constraints_per_item[item2] += 1

# analyze matrix
matrix_size = len(items_of_interest) * len(items_of_interest)
print("dissimilarity matrix: {0} x {0}, {1} entries, {2} are filled (equals {3}%)".format(len(items_of_interest), 
          matrix_size, number_of_filled_entries, 100*(number_of_filled_entries / matrix_size)))

average_num_constraints = 0
for item, num_constraints in constraints_per_item.items():
    print("{0}: {1} constraints".format(item, num_constraints))
    average_num_constraints += num_constraints
print("average number of constraints per item: {0}".format(average_num_constraints / len(items_of_interest)))

# run MDS
plot_coordinates = []
for number_of_dimensions in range(1, args.dims + 1):
    mds = manifold.MDS(n_components=number_of_dimensions, dissimilarity="precomputed", metric=False, n_init = args.n_init, max_iter = args.max_iter, n_jobs = -1)
    results = mds.fit(dissimilarity_matrix)
    
    if args.export != None:
        # need to export
        with open(os.path.join(args.export, "{0}D-vectors.csv".format(number_of_dimensions)), 'w') as f:
            for index, item in enumerate(items_of_interest):
                f.write("{0},{1}\n".format(item, ','.join(map(lambda x: str(x), results.embedding_[index]))))
    
    plot_coordinates.append([number_of_dimensions, results.stress_])
    print(results.stress_)

plt.plot(list(map(lambda x: x[0], plot_coordinates)), list(map(lambda x: x[1], plot_coordinates)), marker='o', linestyle='dashed')
plt.xlabel('number of dimensions')
plt.ylabel('stress')
plt.show()
    
