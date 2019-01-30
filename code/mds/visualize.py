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

parser = argparse.ArgumentParser(description=' Visualizing MDS spaces')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
parser.add_argument('output_folder', help = 'path to the folder where the visualizations should be stored')
parser.add_argument('n', type = int, help = 'number of dimensions of the MDS space')
parser.add_argument('-i', '--image_folder', help = 'the folder containing images of the items', default = None)
parser.add_argument('-z', '--zoom', type = int, help = 'the factor to which the images are scaled', default = 0.15)
args = parser.parse_args()



# read the vectors
vectors = []
with open(args.vector_file, 'r') as f:
    for line in f:
        vector = []
        tokens = line.replace('\n', '').split(',')
        # first entry is the item ID
        vector.append(tokens[0])
        # all other entries are the coordinates
        vector += list(map(lambda x: float(x), tokens[1:]))
        vectors.append(vector)

items = list(map(lambda x: x[0], vectors))

# create a 2D plot for all pairs of dimensions
for first_dim in range(1, args.n + 1):
    for second_dim in range(first_dim + 1, args.n + 1):
        
        x = list(map(lambda x: x[first_dim], vectors))
        y = list(map(lambda x: x[second_dim], vectors))
        
        fig, ax = plt.subplots(figsize=(12,12))
        if args.image_folder != None:
            
            # load all images
            images = []
            for vector in vectors:
                item = vector[0]
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
        
        plt.xlabel('MDS dimension #{0}'.format(first_dim))
        plt.ylabel('MDS dimension #{0}'.format(second_dim))

        output_file_name = os.path.join(args.output_folder, '{0}D-{1}-{2}.png'.format(args.n, first_dim, second_dim))        
        
        fig.savefig(output_file_name, bbox_inches='tight', dpi=200)