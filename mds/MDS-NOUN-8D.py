import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import manifold

# load csv file with distances between images
with open("NOUN_distance_matrix.csv", 'r') as dest_f:
	data_iter = csv.reader(dest_f, delimiter=',', quoting=csv.QUOTE_NONE)
	data = [data for data in data_iter]
	T1 = np.asarray(data)
	# transform items of the array from string to float
	T2 = [[float(letter) for letter in x] for x in T1]
	T3 = np.asarray(T2)
	# transform numbers to decimals [0,1]
	amax = np.amax(T3)
	data_array = T3/amax

# create array with names of images "2001" - "2064"
image_array = list(range(1, 65))
for x in range(0, 9):
	image_array[x] = "200" + str(image_array[x])
for x in range(9, 64):
	image_array[x] = "20" + str(image_array[x])

# Multi-dimensional Scaling (8 dimensions)
mds = manifold.MDS(n_components=8, dissimilarity="precomputed", random_state=None)
results = mds.fit(data_array)
coords = results.embedding_
print(coords)

# write the coordinates in structured format into a csv file
with open("output/8D-vectors.csv", 'w') as f:
    for i in range(64):
        f.write("{0},{1}\n".format(image_array[i], ','.join(map(lambda x: str(x), coords[i]))))


# no plot is created, due to the high-dimensional space
