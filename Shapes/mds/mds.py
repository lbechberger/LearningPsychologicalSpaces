# -*- coding: utf-8 -*-
"""
Runs MDS on the given data set with the given configuration.

Created on Mon Oct 22 14:44:04 2018

@author: lbechberger
"""

import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

if len(sys.argv) < 2:
    raise(Exception("Need at least one argument!"))
input_file_name = sys.argv[1]

# load the data set from the pickle file
with open(input_file_name, "rb") as f:
    data_set = pickle.load(f)

item_ids = list(data_set['items'].keys())
item_names = list(map(lambda x: data_set['items'][x]['name'], item_ids))

# compute dissimilarity matrix
dissimilarity_matrix = np.zeros((len(item_ids), len(item_ids)))

for index1 in range(len(item_ids)):
    for index2 in range(index1 + 1, len(item_ids)):
        
        item1 = item_ids[index1]
        item2 = item_ids[index2]

        tuple_id = str(sorted([item1, item2]))
        if tuple_id not in data_set['similarities'].keys():
            # unknown similarities: simply leave zeroes in there (are treated as "unknown" by MDS algorithm)
            # see https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/manifold/mds.py#L100
            continue
        
        # compute dissimilarity
        similarity_ratings = data_set['similarities'][tuple_id]['values']
        overall_similarity = np.mean(similarity_ratings)
        dissimilarity = 5 - overall_similarity
        
        # add to matrix
        dissimilarity_matrix[index1][index2] = dissimilarity
        dissimilarity_matrix[index2][index1] = dissimilarity

# run MDS
plot_coordinates = []
for number_of_dimensions in range(1, 21):
    mds = manifold.MDS(n_components=number_of_dimensions, dissimilarity="precomputed", random_state=None, metric=False, n_init = 64, max_iter = 5000)
    results = mds.fit(dissimilarity_matrix)
    
    plot_coordinates.append([number_of_dimensions, results.stress_])
    print(results.stress_)

plt.plot(list(map(lambda x: x[0], plot_coordinates)), list(map(lambda x: x[1], plot_coordinates)), marker='o', linestyle='dashed')
plt.xlabel('number of dimensions')
plt.ylabel('stress')
plt.show()
    
