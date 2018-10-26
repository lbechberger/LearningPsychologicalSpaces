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

entries_filled = 0
entries_open = 0

for index1 in range(len(item_ids)):
    for index2 in range(index1 + 1, len(item_ids)):
        
        item1 = item_ids[index1]
        item2 = item_ids[index2]

        # compute dissimilarity
        tuple_id = str(sorted([item1, item2]))
        if tuple_id not in data_set['similarities'].keys():
            # unknown similarities: simply leave zeroes in there
            entries_open += 1
            continue
        
        similarity_ratings = data_set['similarities'][tuple_id]['values']
        overall_similarity = np.mean(similarity_ratings)
        dissimilarity = 5 - overall_similarity
        
        # add to matrix
        dissimilarity_matrix[index1][index2] = dissimilarity
        dissimilarity_matrix[index2][index1] = dissimilarity
        entries_filled += 1

print(entries_filled, entries_open)

# Multi-dimensional Scaling (2 dimensions)
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=None, metric=False)
results = mds.fit(dissimilarity_matrix)
coords = results.embedding_
print(results.stress_)

# write the coordinates in structured format into a csv file
with open("2D-vectors.csv", 'w') as f:
    for i in range(len(item_ids)):
        f.write("{0},{1}\n".format(item_ids[i], ','.join(map(lambda x: str(x), coords[i]))))

# plot first two dimensions (o)
plt.subplots_adjust(bottom=0.1)
plt.scatter(
	coords[:, 0], coords[:, 1], marker='o'
)
for label, x, y in zip(item_names, coords[:, 0], coords[:, 1]):
	plt.annotate(
		label,
		xy=(x, y), xytext=(-20, 20),
		textcoords='offset points', ha='right', va='bottom',
		bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
		arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()
