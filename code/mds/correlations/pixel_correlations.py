# -*- coding: utf-8 -*-
"""
Computes the different similarity measures  between the images on a pixel basis.
Compares these similarities to the human similarity ratings.

Created on Tue Dec  4 09:27:06 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from code.util import precompute_distances, compute_correlations, distance_functions
from code.util import downscale_image, aggregator_functions, load_image_files_pixel
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args

parser = argparse.ArgumentParser(description='Pixel-based similarity baseline')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('distance_file', help = 'the pickle file containing the pre-computed distances')
parser.add_argument('output_file', help = 'the csv file to which the output should be saved')
parser.add_argument('-i', '--image_folder', help = 'the folder containing the original images', default = None)
parser.add_argument('-w', '--width', type = int, default = 300, help = 'the width of the image, used to determine the maximal block size')
parser.add_argument('-g', '--greyscale', action = 'store_true', help = 'only consider greyscale information (i.e., collapse color channels)')
parser.add_argument('-n', '--n_folds', type = int, help = 'number of folds to use for cross-validation when optimizing weights', default = 5)
parser.add_argument('-s', '--seed', type = int, help = 'fixed seed to use for creating the folds', default = None)
add_correlation_metrics_to_parser(parser)
args = parser.parse_args()

correlation_metrics = get_correlation_metrics_from_args(args)

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

items = input_data['items']
target_dissimilarities = input_data['dissimilarities']

if args.image_folder is not None:
    # load images and compute distances from scratch
    images = load_image_files_pixel(items, args.image_folder)
    distances = {}
else:
    # just load distances from file
    with open(args.distance_file, 'rb') as f_in:
        distances = pickle.load(f_in)

with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("aggregator,block_size,image_size,scoring,weights,{0}\n".format(','.join(correlation_metrics)))
    last_image_size = 9999
    for block_size in range(1, args.width + 1):
        
        # if the resulting image size is the same: skip this block size (will just introduce more noise due to zero padding)
        current_image_size = int(np.ceil(args.width / block_size))
        if current_image_size == last_image_size:
            continue
        else:
            last_image_size = current_image_size

        for aggregator_name, aggregator_function in aggregator_functions.items():
        
            if block_size == 1 and aggregator_name in ['std', 'var', 'mean', 'min', 'median', 'product']:
                # can't compute std or var on a single element
                # value for all others is identical to max, so only compute once
                continue
            
            if args.image_folder is not None:
                # transform images for distance computation
                transformed_images = []
                for img in images:
                    transformed_img, image_size = downscale_image(img, aggregator_function, block_size, args.greyscale, (1,-1))
                    transformed_images.append(transformed_img)
                
    
            for distance_function in sorted(distance_functions.keys()):

                if args.image_folder is not None:
                    # precompute distances based on transformed images and store them for later output
                    precomputed_distances, precomputed_targets = precompute_distances(transformed_images, target_dissimilarities, distance_function)
                    if block_size not in distances:
                        distances[block_size] = {}
                    if aggregator_name not in distances[block_size]:
                        distances[block_size][aggregator_name] = {}
                    distances[block_size][aggregator_name][distance_function] = (precomputed_distances, precomputed_targets)
                else:
                    # simply grab them from the loaded dictionary
                    precomputed_distances, precomputed_targets = distances[block_size][aggregator_name][distance_function]

                # raw correlations
                correlation_results = compute_correlations(precomputed_distances, precomputed_targets, distance_function)    
                f_out.write("{0},{1},{2},{3},fixed,{4}\n".format(aggregator_name, block_size, image_size, distance_function, 
                                                                    ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))
                # correlation with optimized weights
            
                correlation_results = compute_correlations(precomputed_distances, precomputed_targets, distance_function, args.n_folds, args.seed)    
                f_out.write("{0},{1},{2},{3},optimized,{4}\n".format(aggregator_name, block_size, image_size, distance_function, 
                                                                    ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))
                
                print('done with {0}-{1}-{2}; weights: {3}'.format(block_size, aggregator_name, distance_function, correlation_results['weights']))

# output the collected distances if necessary
if args.image_folder is not None:
    with open(args.distance_file, 'wb') as f_out:
        pickle.dump(distances, f_out)