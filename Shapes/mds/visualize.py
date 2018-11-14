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

parser = argparse.ArgumentParser(description='MDS for shapes')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
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
x = list(map(lambda x: x[1], vectors))
y = list(map(lambda x: x[2], vectors))

fix, ax = plt.subplots()
if args.image_folder != None:
    
    # load all images
    images = []
    for vector in vectors:
        item = vector[0]
        for file_name in os.listdir(args.image_folder):
            if os.path.isfile(os.path.join(args.image_folder, file_name)) and item in file_name:
                # found the corresponding image
                images.append(plt.imread(os.path.join(args.image_folder, file_name)))
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
    ax.scatter(x,y)
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

plt.show()