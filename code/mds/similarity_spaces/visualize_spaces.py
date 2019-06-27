# -*- coding: utf-8 -*-
"""
Visualizes a given MDS space. Only works for spaces of dimensionality > 1.

Created on Wed Nov 14 09:54:42 2018

@author: lbechberger
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from code.util import load_mds_vectors

parser = argparse.ArgumentParser(description=' Visualizing MDS spaces')
parser.add_argument('vector_folder', help = 'path to the folder containing the vectors')
parser.add_argument('output_folder', help = 'path to the folder where the visualizations should be stored')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = int, help = 'the factor to which the images are scaled', default = 0.15)
parser.add_argument('-m', '--max', type = int, help = 'size of the largest space to be visualized', default = 10)
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
if args.image_folder != None:
    
    images = []
    items = list(sorted(list(vector_map.values())[0].keys()))
    for item in items:
        for file_name in os.listdir(args.image_folder):
            if os.path.isfile(os.path.join(args.image_folder, file_name)) and item in file_name:
                # found the corresponding image
                img = Image.open(os.path.join(args.image_folder, file_name), 'r')
                img = img.convert("RGBA")
                
                # conversion of white to alpha based on https://stackoverflow.com/a/765774
                img_data = img.getdata()
                new_data = []
                for item in img_data:
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                img.putdata(new_data)
                images.append(img)
                break

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
            
            fig, ax = plt.subplots(figsize=(12,12))
            if args.image_folder != None:
                
                # plot scatter plot with images        
                # based on https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
                if ax is None:
                    ax = plt.gca()
                x, y = np.atleast_1d(x, y)
                artists = []
                for x0, y0, im0 in zip(x, y, images):
                    im = OffsetImage(im0, zoom=args.zoom)
                    ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                    artists.append(ax.add_artist(ab))
                ax.update_datalim(np.column_stack([x, y]))
                ax.autoscale()
                ax.scatter(x,y, s=0)
            else:
                # plot scatter plot without images, but with item IDs instead
                ax.scatter(x,y)
                for label, x0, y0 in zip(items, x, y):
                    plt.annotate(
            		label,
            		xy=(x0, y0), xytext=(-20, 20),
            		textcoords='offset points', ha='right', va='bottom',
            		bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            		arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)
            ax.set_xlabel('MDS dimension #{0}'.format(first_dim + 1), fontsize = 20)
            ax.set_ylabel('MDS dimension #{0}'.format(second_dim + 1), fontsize = 20)
    
            output_file_name = os.path.join(args.output_folder, '{0}D-{1}-{2}.png'.format(dim, first_dim + 1, second_dim + 1))        
            
            fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
            plt.close()