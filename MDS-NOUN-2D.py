import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import manifold

# load csv file with distances between images
with open("D:/User/Downloads/Data-Sets/NOUN_distance_matrix.csv", 'r') as dest_f:
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

# Multi-dimensional Scaling (2 dimensions)
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=None)
results = mds.fit(data_array)
coords = results.embedding_
print(coords)

# plot first two dimensions (o)
plt.subplots_adjust(bottom=0.1)
plt.scatter(
	coords[:, 0], coords[:, 1], marker='o'
)
for label, x, y in zip(image_array, coords[:, 0], coords[:, 1]):
	plt.annotate(
		label,
		xy=(x, y), xytext=(-20, 20),
		textcoords='offset points', ha='right', va='bottom',
		bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
		arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()
