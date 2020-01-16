# -*- coding: utf-8 -*-
"""
Visualizes a given MDS space. Only works for spaces of dimensionality > 1.

Created on Wed Nov 14 09:54:42 2018

@author: lbechberger
"""

import argparse, os
from code.util import load_mds_vectors, load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description=' Visualizing MDS spaces')
parser.add_argument('vector_folder', help = 'path to the folder containing the vectors')
parser.add_argument('output_folder', help = 'path to the folder where the visualizations should be stored')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-m', '--max', type = int, help = 'size of the largest space to be visualized', default = 10)
parser.add_argument('-d', '--directions_file', help = 'file containing the directions to plot', default = None)
args = parser.parse_args()

# first read in all the vectors
vector_map = {}
for file_name in os.listdir(args.vector_folder):
    tokens = file_name.split('D-vectors')
    if len(tokens) > 1:
        # valid file name
        dim = int(tokens[0])
        
        # skip space if it is too large
        if dim > args.max:
            continue

        local_dict = load_mds_vectors(os.path.join(args.vector_folder, file_name))
        vector_map[dim] = local_dict
          

# then read in all the images
items = list(sorted(list(vector_map.values())[0].keys()))
    
images = None
if args.image_folder != None:
    images = load_item_images(args.image_folder, items)    

# if we have directions: read them in
if args.directions_file is not None:
    directions = {}
    # TODO
    
# iterate over all spaces
for dim, mapping in vector_map.items():
    print('        {0} dimensions'.format(dim))
    items = list(sorted(mapping.keys()))
    vectors = []
    for item in items:
        vectors.append(mapping[item])
    # create a 2D plot for all pairs of dimensions
    for first_dim in range(dim):
        for second_dim in range(first_dim + 1, dim):
            
            x = list(map(lambda x: x[first_dim], vectors))
            y = list(map(lambda x: x[second_dim], vectors))
            output_file_name = os.path.join(args.output_folder, '{0}D-{1}-{2}.png'.format(dim, first_dim + 1, second_dim + 1))        
            x_label = 'MDS dimension #{0}'.format(first_dim + 1)
            y_label = 'MDS dimension #{0}'.format(second_dim + 1)
            
            create_labeled_scatter_plot(x, y, output_file_name, x_label = x_label, y_label = y_label, images = images, zoom = args.zoom, item_ids = items)            