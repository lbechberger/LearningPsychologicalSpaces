# -*- coding: utf-8 -*-
"""
Computes the correlations between the scales for two interpretable dimensions.

Created on Mon Jan 27 15:52:20 2020

@author: lbechberger
"""

import pickle, argparse, os
from scipy.stats import spearmanr
from code.util import load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description='Correlating the scales of two dimensions')
parser.add_argument('first_dimension', help = 'pickle file containing the data about the first dimension')
parser.add_argument('second_dimension', help = 'pickle file containing the data about the second dimension')
parser.add_argument('output_folder', help = 'folder where the plots will be stored')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-f', '--first_name', help = 'name for the first dimension', default = 'first')
parser.add_argument('-s', '--second_name', help = 'name for the second dimension', default = 'second')
args = parser.parse_args()

# load dimension data
with open(args.first_dimension, 'rb') as f_in:
    first_data = pickle.load(f_in)
with open(args.second_dimension, 'rb') as f_in:
    second_data = pickle.load(f_in)

# sorted list of item_ids
items_sorted = list(sorted(first_data['binary'].keys()))

# then read in all the images
images = None
if args.image_folder != None:
    images = load_item_images(args.image_folder, items_sorted)    


# look at binary, continuous, RT separately
for scale_type in sorted(first_data.keys()):
    first_scale = first_data[scale_type]
    second_scale = second_data[scale_type]
    
    item_ids = sorted(first_scale.keys())
    first_values = [first_scale[item_id] for item_id in item_ids]
    second_values = [second_scale[item_id] for item_id in item_ids]
    
    spearman, _ = spearmanr(first_values, second_values)
    print("Spearman correlation for {0} scale: {1}".format(scale_type, spearman))
    
    # create scatter plot
    output_file_name = os.path.join(args.output_folder, 'scatter-{0}-{1}-{2}.png'.format(args.first_name, args.second_name, scale_type))        
    create_labeled_scatter_plot(first_values, second_values, output_file_name, x_label = args.first_name, y_label = args.second_name, images = images, zoom = args.zoom)  

    
    