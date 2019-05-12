# -*- coding: utf-8 -*-
"""
Analyzes whether the feature vectors show strong clustering tendencies, using the Silhouette Coefficient.

Created on Fri May 10 10:17:30 2019

@author: lbechberger
"""

import pickle, argparse, random
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser(description='Preparing the target vectors for the regression')
parser.add_argument('input_file', help = 'csv file containing the paths to target vector files')
parser.add_argument('-n', '--n_sample', type = int, help = 'sample size for each original image', default = 100)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

if args.seed is not None:
   random.seed(args.seed)

with open(args.input_file, 'rb') as f:
    data_set = pickle.load(f)

vectors = []
labels = []

for image_name, feature_vectors in data_set.items():
    vecs = feature_vectors
    random.shuffle(vecs)
    vectors += vecs[:args.n_sample]
    labels += [image_name]*args.n_sample

shuffled_vectors = list(vectors)
random.shuffle(shuffled_vectors)

print('ACTUAL RESULTS')
silhouette_euclidean = silhouette_score(vectors, labels, metric = 'euclidean')
print('Euclidean:', silhouette_euclidean)
silhouette_manhattan = silhouette_score(vectors, labels, metric = 'manhattan')
print('Manhattan:', silhouette_manhattan)
silhouette_cosine = silhouette_score(vectors, labels, metric = 'cosine')
print('Cosine:', silhouette_cosine)

print('SHUFFLED RESULTS')
silhouette_euclidean = silhouette_score(shuffled_vectors, labels, metric = 'euclidean')
print('Euclidean:', silhouette_euclidean)
silhouette_manhattan = silhouette_score(shuffled_vectors, labels, metric = 'manhattan')
print('Manhattan:', silhouette_manhattan)
silhouette_cosine = silhouette_score(shuffled_vectors, labels, metric = 'cosine')
print('Cosine:', silhouette_cosine)