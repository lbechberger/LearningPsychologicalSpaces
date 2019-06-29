# -*- coding: utf-8 -*-
"""
Computes the different similarity measures  between the images on a pixel basis.
Compares these similarities to the human similarity ratings.

Created on Tue Dec  4 09:27:06 2018

@author: lbechberger
"""

import pickle, argparse
import numpy as np
from code.util import compute_correlations, distance_functions, downscale_image, aggregator_functions, load_image_files_pixel

parser = argparse.ArgumentParser(description='Pixel-based similarity baseline')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_file', help = 'the csv file to which the output should be saved', default='pixel.csv')
parser.add_argument('-s', '--size', type = int, default = 300, help = 'the size of the image, used to determine the maximal block size')
parser.add_argument('-g', '--greyscale', action = 'store_true', help = 'only consider greyscale information (i.e., collapse color channels)')
args = parser.parse_args()

# load the real similarity data
with open(args.similarity_file, 'rb') as f:
    input_data = pickle.load(f)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

images = load_image_files_pixel(item_ids, args.image_folder, )

with open(args.output_file, 'w', buffering=1) as f:

    f.write("aggregator,block_size,image_size,scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")
    last_image_size = 9999
    for block_size in range(1, args.size + 1):
        
        # if the resulting image size is the same: skip this block size (will just introduce more noise due to zero padding)
        current_image_size = int(np.ceil(args.size / block_size))
        if current_image_size == last_image_size:
            continue
        else:
            last_image_size = current_image_size
            print('    {0}'.format(current_image_size))

        for aggregator_name, aggregator_function in aggregator_functions.items():
        
            if block_size == 1 and aggregator_name in ['std', 'var', 'mean', 'min', 'median', 'product']:
                # can't compute std or var on a single element
                # value for all others is identical to max, so only compute once
                continue
            
            transformed_images = []
            for img in images:
                transformed_img, image_size = downscale_image(img, aggregator_function, block_size, args.greyscale, (1,-1))
                transformed_images.append(transformed_img)
    
            for distance_name, distance_function in distance_functions.items():

                correlation_metrics = compute_correlations(transformed_images, target_dissimilarities, distance_function)
                
                f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(aggregator_name, block_size, image_size, distance_name, 
                                                                    correlation_metrics['pearson'], 
                                                                    correlation_metrics['spearman'], 
                                                                    correlation_metrics['kendall'], 
                                                                    correlation_metrics['r2_linear'], 
                                                                    correlation_metrics['r2_isotonic']))