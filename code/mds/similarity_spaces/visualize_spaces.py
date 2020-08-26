# -*- coding: utf-8 -*-
"""
Visualizes a given MDS space. Only works for spaces of dimensionality > 1.

Created on Wed Nov 14 09:54:42 2018

@author: lbechberger
"""

import argparse, os, csv, pickle
import numpy as np
from code.util import load_item_images, create_labeled_scatter_plot

parser = argparse.ArgumentParser(description=' Visualizing MDS spaces')
parser.add_argument('vector_file', help = 'path to the pickle file containing the vectors')
parser.add_argument('output_folder', help = 'path to the folder where the visualizations should be stored')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = float, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-m', '--max', type = int, help = 'size of the largest space to be visualized', default = 10)
parser.add_argument('-d', '--directions_file', help = 'file containing the directions to plot', default = None)
parser.add_argument('-c', '--criterion', help = 'filtering criterion to use for the directions to plot', default = 'kappa')
parser.add_argument('-r', '--regions', action = 'store_true', help = 'plot conceptual regions')
args = parser.parse_args()

# first read in all the vectors
with open(args.vector_file, 'rb') as f_in:
    vector_map = pickle.load(f_in)

# then read in all the images
items = list(sorted(list(vector_map.values())[0].keys()))
    
images = None
if args.image_folder != None:
    images = load_item_images(args.image_folder, items)    

# if we have directions: read them in
if args.directions_file is not None:
    directions = {}
    for i in range(1, args.max + 1):
        directions[i] = {}
    
    with open(args.directions_file, 'r') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            dims = int(row['dims'])
            criterion = row['criterion']
            if dims > args.max or criterion != args.criterion:
                continue            
            direction_name = row['direction_name']
            vector = [float(row['d{0}'.format(d)]) for d in range(dims)]
            directions[dims][direction_name] = vector
    
# iterate over all spaces
for dim in range(1, args.max + 1):
    mapping = vector_map[dim]
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
            
            if args.directions_file is not None:
                plot_directions = {}
                for direction_name, direction_vector in directions[dim].items():
                    plot_directions[direction_name] = [direction_vector[first_dim], direction_vector[second_dim]]
            else:
                plot_directions = None
            
            if args.regions:
                regions = []
                colors = {'VV': ['lightcoral', 'indianred', 'salmon', 'tomato', 'firebrick', 'orangered'], 
                          'VC': ['lightgreen', 'darkseagreen', 'forestgreen', 'mediumspringgreen', 'limegreen', 'green']}
                for category, category_dict in vector_map['categories'].items():
                    region = []
                    for item in category_dict['items']:
                        vector = [x[items.index(item)], y[items.index(item)]]
                        region.append(vector)
                    linestyle = 'dashed' if category_dict['visSim'] == 'VC' else 'dotted'
                    color = colors[category_dict['visSim']].pop()
                    regions.append((np.array(region), color, linestyle))
            else:
                regions = None
            
            create_labeled_scatter_plot(x, y, output_file_name, x_label = x_label, y_label = y_label, 
                                        images = images, zoom = args.zoom, item_ids = items, 
                                        directions = plot_directions, regions = regions)            