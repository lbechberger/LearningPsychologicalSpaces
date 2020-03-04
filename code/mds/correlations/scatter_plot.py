# -*- coding: utf-8 -*-
"""
Creates a scatter plot of estimated distances versus human dissimilarity ratings.

Created on Wed Jun 19 00:38:28 2019

@author: lbechberger
"""

import matplotlib.pyplot as plt
import argparse, pickle, os
from code.util import compute_correlations, extract_inception_features, downscale_images, aggregator_functions, load_image_files_pixel, load_image_files_ann, load_mds_vectors

parser = argparse.ArgumentParser(description='Scatter plot generation')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('output_file', help = 'path to the output image file')
parser.add_argument('--distance', '-d', help = 'distance function to use', default = 'Euclidean')
parser.add_argument('--mds', '-m', help = 'path to the MDS space to use', default = None)
parser.add_argument('--ann', '-a', help = 'path to ANN model file', default = None)
parser.add_argument('--pixel', '-p', help = 'aggregator for pixel baseline', default = None)
parser.add_argument('--features', '-f', help = 'path to feature folder', default = None)
parser.add_argument('--block_size', '-b', type = int, help = 'block size for pixel baseline', default = 1)
parser.add_argument('--image_folder', '-i', help = 'path to image folder', default = '.')
parser.add_argument('--greyscale', '-g', action = 'store_true', help = 'should images be converted to greyscale for pixel baseline?')
parser.add_argument('--space', help = 'dimensions to use (separated by dash)', default = 'FORM')
parser.add_argument('--type', '-t', help = 'feature type to use', default = 'attentive')
parser.add_argument('--optimized', '-o', action = 'store_true', help = 'if this flag is set, weights are optimized in cross-validation')
parser.add_argument('-n', '--n_folds', type = int, help = 'number of folds to use for cross-validation when optimizing weights', default = 5)
parser.add_argument('-s', '--seed', type = int, help = 'fixed seed to use for creating the folds', default = None)
args = parser.parse_args()

if sum([args.mds is not None, args.ann is not None, args.pixel is not None, args.features is not None]) != 1:
    raise Exception('Must use exactly one prediction type!')

with open(args.similarity_file, 'rb') as f:
    input_data = pickle.load(f)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

if args.mds is not None:
    transformed_items = load_mds_vectors(args.mds, item_ids)
    x_label = '{0} Distance in MDS Space'.format(args.distance)
    
elif args.ann is not None:
    images = load_image_files_ann(item_ids, args.image_folder)
    transformed_items = extract_inception_features(images, args.ann, (1, -1))
    x_label = '{0} Distance of ANN Activation Vectors'.format(args.distance)

elif args.features is not None:
    
    feature_names = args.space.split('-')
    feature_data = {}
    for feature_name in feature_names:
        file_name = os.path.join(args.features, '{0}.pickle'.format(feature_name))
        with open(file_name, 'rb') as f_in:
            local_data = pickle.load(f_in)
        
        scale_values = []
        for item_id in item_ids:
            if args.type in local_data.keys():
                scale_values.append(local_data[args.type][item_id])
            else:
                scale_values.append(local_data['metadata'][item_id])
            
        feature_data[feature_name] = scale_values
    
        
    transformed_items = []
    for i in range(len(item_ids)):
        item_vec = []
        for feature_name in feature_names:
            item_vec.append(feature_data[feature_name][i])
        transformed_items.append(item_vec)
    x_label = '{0} Distance based on Shape Features'.format(args.distance)
    
else: # i.e., args.pixel is not None
    images = load_image_files_pixel(item_ids, args.image_folder)
    transformed_items = downscale_images(images, aggregator_functions[args.pixel], args.block_size, args.greyscale, (1,-1))
    x_label = '{0} Distance of Downscaled Images'.format(args.distance)
    
if args.optimized:
    folds = args.n_folds
    seed = args.seed
else:
    folds = None
    seed = None

corr_dict = compute_correlations(transformed_items, target_dissimilarities, args.distance, folds, seed)

fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.scatter(corr_dict['predictions'], corr_dict['targets'])
plt.xlabel(x_label, fontsize = 20)
plt.ylabel('Dissimilarity from Psychological Study', fontsize = 20)


fig.savefig(args.output_file, bbox_inches='tight', dpi=200)
plt.close()
