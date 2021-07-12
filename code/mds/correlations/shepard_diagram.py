# -*- coding: utf-8 -*-
"""
Creates a shepard diagram of estimated distances versus human dissimilarity ratings.

Created on Wed Jun 19 00:38:28 2019

@author: lbechberger
"""

import matplotlib.pyplot as plt
import argparse, pickle, os
from sklearn.isotonic import IsotonicRegression
from code.util import compute_correlations


parser = argparse.ArgumentParser(description='Scatter plot generation')
parser.add_argument('similarity_file', help = 'the input file containing the target dissimilarity ratings')
parser.add_argument('distance_file', help = 'the input file containing the pre-computed distances and dissimilarities')
parser.add_argument('output_file', help = 'path to the output image file')
parser.add_argument('--distance', '-d', help = 'distance function to use', default = 'Euclidean')
parser.add_argument('--mds', '-m', type = int, help = 'dimensionality of the MDS space', default = None)
parser.add_argument('--ann', '-a', action = 'store_true', help = 'use ANN baseline')
parser.add_argument('--features', '-f', help = 'name of feature space to use', default = None)
parser.add_argument('--type', '-t', help = 'feature type to use', default = 'attentive')
parser.add_argument('--pixel', '-p', help = 'aggregator for pixel baseline', default = None)
parser.add_argument('--block_size', '-b', type = int, help = 'block size for pixel baseline', default = 1)
parser.add_argument('--random', '-r', type = int, help = 'dimensionality of the random baseline space', default = None)
parser.add_argument('--distribution', help = 'distribution type to use', default = 'normal')
parser.add_argument('--optimized', '-o', action = 'store_true', help = 'if this flag is set, weights are optimized in cross-validation')
parser.add_argument('--similarity_name', help = 'type of similarity matrix being used', default = 'Mean')
parser.add_argument('-n', '--n_folds', type = int, help = 'number of folds to use for cross-validation when optimizing weights', default = 5)
parser.add_argument('-s', '--seed', type = int, help = 'fixed seed to use for creating the folds', default = None)
args = parser.parse_args()

if sum([args.mds is not None, args.ann, args.pixel is not None, args.features is not None, args.random is not None]) != 1:
    raise Exception('Must use exactly one prediction type!')

with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

# special treatment for pixel baseline since we have folder of individual files
if args.pixel is not None:
    input_file_name = os.path.join(args.distance_file, '{0}-{1}-{2}.pickle'.format(args.block_size, args.pixel, args.distance))
    with open(input_file_name, 'rb') as f_in:
        precomputed_distances = pickle.load(f_in)
    x_label = '{0} Distance of Downscaled Images\n(Block Size {1}, Aggregation with {2})'.format(args.distance, args.block_size, args.pixel)
else:
    with open(args.distance_file, 'rb') as f_in:
        distances = pickle.load(f_in)
    if args.mds is not None:
        precomputed_distances = distances['MDS'][args.mds][args.distance][0]
        x_label = '{0} Distance in {1}-Dimensional MDS Space'.format(args.distance, args.mds)
    elif args.ann:
        precomputed_distances = distances[args.distance]
        x_label = '{0} Distance of ANN Activation Vectors'.format(args.distance)
    elif args.features is not None:
        precomputed_distances = distances[args.features][args.type][args.distance]
        x_label = '{0} Distance Based on {1} Shape Features:\n{2}'.format(args.distance, args.type, args.features)
    elif args.random is not None:
        precomputed_distances = distances[args.distribution][args.random][args.distance][0]
        x_label = '{0} Distance in a Random {1}-Dimensional Space with {2} Distribution'.format(args.distance, args.random, args.distribution)
    else: # args.pixel is not None:
        raise(Exception('This should not happen!'))

target_dissimilarities = input_data['dissimilarities']

if args.optimized:
    folds = args.n_folds
    seed = args.seed
else:
    folds = None
    seed = None

corr_dict = compute_correlations(precomputed_distances, target_dissimilarities, args.distance, folds, seed)

fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)

x = corr_dict['predictions'].reshape(-1)
y = corr_dict['targets'].reshape(-1)

ax.scatter(x, y)
plt.xlabel(x_label, fontsize = 20)
plt.ylabel('{0} Dissimilarity from Psychological Study'.format(args.similarity_name), fontsize = 20)

ir = IsotonicRegression()
ir.fit(x, y)
best_fit = ir.predict(x)
xy_sorted = sorted(zip(x,best_fit), key = lambda x: x[0])
x_new, y_new = ([i for i, j in xy_sorted],[j for i,j in xy_sorted])
ax.plot(x_new, y_new, 'r--', linewidth = 4)

fig.savefig(args.output_file, bbox_inches='tight', dpi=200)
plt.close()
