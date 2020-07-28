# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:37:53 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from code.util import load_item_images

parser = argparse.ArgumentParser(description='Plotting similarity tables')
parser.add_argument('visual_similarity_file', help = 'the input file containing the visual similarity ratings')
parser.add_argument('conceptual_similarity_file', help = 'the input file containing the conceptual similarity ratings')
parser.add_argument('output_folder', help = 'the folder where the plots shall be stored')
parser.add_argument('-i', '--image_folder', help = 'the folder containing the original images', default = None)
args = parser.parse_args()

np.random.seed(42) # fixed random seed to ensure reproducibility

# load the similarity data
with open(args.visual_similarity_file, 'rb') as f_in:
    visual_input_data = pickle.load(f_in)
with open(args.conceptual_similarity_file, 'rb') as f_in:
    conceptual_input_data = pickle.load(f_in)

# load the images (if applicable)
images = None
if args.image_folder is not None:
    images = load_item_images(args.image_folder, list(visual_input_data['items']))

# does the actual plotting
def plot_matrices(visual_matrix, conceptual_matrix, legend, output_file_name, mode='merge', images = None):

    # helper function for plotting images instead of item names
    def offset_image(coord, ax):
        img = images[coord]
        im = OffsetImage(img, zoom = 0.05)
        im.image.axes = ax
        
        ab = AnnotationBbox(im, (0, coord), xybox=(-16., -0.), frameon=False, xycoords='data', 
                            boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    unified_matrix = np.ones(visual_matrix.shape)
    for i in range(len(legend)):
        for j in range(len(legend)):
            if i <= j and mode in ['merge','visual']:
                unified_matrix[i][j] = visual_matrix[i][j]
            if i >= j and mode in ['merge','conceptual']:
                unified_matrix[i][j] = conceptual_matrix[i][j]

    if mode == 'merge':
        figsize = (12,9)
    else:
        figsize = (6.4,4.8)
    fig, ax = plt.subplots(figsize=figsize)#figsize=(width,width))
    im = ax.imshow(unified_matrix, cm.get_cmap('gray_r'))
    
    if mode == 'visual':
        ax.tick_params(bottom=False,left=False,top=True,right=True, labeltop=True, labelright=True, labelbottom=False, labelleft=False)    
        plt.setp(ax.get_xticklabels(), rotation='vertical')
    else:
        plt.setp(ax.get_xticklabels(), rotation='vertical')
        cbar = ax.figure.colorbar(im, ticks=[1.1,4.9], ax=ax, orientation='vertical')
        cbar.ax.set_ylabel("similarity", rotation='vertical')
        cbar.ax.set_yticklabels(['low','high'])


    if images is None:
        ax.set_xticks(np.arange(len(legend)))
        ax.set_yticks(np.arange(len(legend)))
        ax.set_xticklabels(legend)
        ax.set_yticklabels(legend) 
    else:
        for ctr in range(len(legend)):
            offset_image(ctr, ax)
    

    fig.tight_layout()     
    fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
    #plt.show()

# first plot item-based matrices
visual_item_matrix = visual_input_data['similarities']
conceptual_item_matrix = conceptual_input_data['similarities']
file_name = os.path.join(args.output_folder, 'heatmap_item_based.png')
plot_matrices(visual_item_matrix, conceptual_item_matrix, visual_input_data['item_names'], file_name, images=images)

# now plot category-based matrices
category_names = visual_input_data['category_names']
visual_category_matrix = np.zeros((len(category_names), len(category_names)))
conceptual_category_matrix = np.zeros((len(category_names), len(category_names)))

items_per_category = int(visual_item_matrix.shape[0]/len(category_names))

for first_cat in range(len(category_names)):
    for second_cat in range(first_cat, len(category_names)):
        
        visual_values = []
        conceptual_values = []
        for first_item in range(items_per_category):
            for second_item in range(first_item + 1, items_per_category):
                x = first_cat * items_per_category + first_item
                y = second_cat * items_per_category + second_item
                visual_values.append(visual_item_matrix[x][y])
                conceptual_values.append(conceptual_item_matrix[x][y])
        
        visual_median = np.median(visual_values)
        visual_category_matrix[first_cat][second_cat] = visual_median
        visual_category_matrix[second_cat][first_cat] = visual_median

        conceptual_median = np.median(conceptual_values)
        conceptual_category_matrix[first_cat][second_cat] = conceptual_median
        conceptual_category_matrix[second_cat][first_cat] = conceptual_median

visual_file_name = os.path.join(args.output_folder, 'heatmap_category_based_visual.png')
conceptual_file_name = os.path.join(args.output_folder, 'heatmap_category_based_conceptual.png')
plot_matrices(visual_category_matrix, conceptual_category_matrix, category_names, visual_file_name, mode='visual')
plot_matrices(visual_category_matrix, conceptual_category_matrix, category_names, conceptual_file_name, mode='conceptual')