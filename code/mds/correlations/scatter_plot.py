# -*- coding: utf-8 -*-
"""
Creates a scatter plot of estimated distances versus human dissimilarity ratings.

Created on Wed Jun 19 00:38:28 2019

@author: lbechberger
"""

import matplotlib.pyplot as plt
import argparse, pickle
from code.util import compute_correlations, distance_functions, extract_inception_features, downscale_images, aggregator_functions, load_image_files_pixel, load_image_files_ann, load_mds_vectors

parser = argparse.ArgumentParser(description='Scatter plot generation')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('output_file', help = 'path to the output image file')
parser.add_argument('--distance', '-d', help = 'distance function to use', default = 'Euclidean')
parser.add_argument('--mds', '-m', help = 'path to the MDS space to use', default = None)
parser.add_argument('--ann', '-a', help = 'path to ANN model file', default = None)
parser.add_argument('--pixel', '-p', help = 'aggregator for pixel baseline', default = None)
parser.add_argument('--block_size', '-b', type = int, help = 'block size for pixel baseline', default = 1)
parser.add_argument('--image_folder', '-i', help = 'path to image folder', default = '.')
parser.add_argument('--greyscale', '-g', action = 'store_true', help = 'should images be converted to greyscale for pixel baseline?')
args = parser.parse_args()

if sum([args.mds is not None, args.ann is not None, args.pixel is not None]) != 1:
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
    
else: # i.e., args.pixel is not None
    images = load_image_files_pixel(item_ids, args.image_folder)
    transformed_items = downscale_images(images, aggregator_functions[args.pixel], args.block_size, args.greyscale, (1,-1))
    x_label = '{0} Distance of Downscaled Images'.format(args.distance)
    

corr_dict = compute_correlations(transformed_items, target_dissimilarities, distance_functions[args.distance])

fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.scatter(corr_dict['predictions'], corr_dict['targets'])
plt.xlabel(x_label, fontsize = 20)
plt.ylabel('Dissimilarity from Psychological Study', fontsize = 20)


fig.savefig(args.output_file, bbox_inches='tight', dpi=200)
plt.close()
