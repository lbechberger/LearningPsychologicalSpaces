# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:37:53 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm

parser = argparse.ArgumentParser(description='Plotting similarity tables')
parser.add_argument('visual_similarity_file', help = 'the input file containing the visual similarity ratings')
parser.add_argument('conceptual_similarity_file', help = 'the input file containing the conceptual similarity ratings')
parser.add_argument('output_folder', help = 'the folder where the plots will be stored')
args = parser.parse_args()


# load the similarity data
with open(args.visual_similarity_file, 'rb') as f_in:
    visual_input_data = pickle.load(f_in)
with open(args.conceptual_similarity_file, 'rb') as f_in:
    conceptual_input_data = pickle.load(f_in)

# merge the two given matrices
def merge_matrices(conceptual_matrix, visual_matrix, diagonal_from_second = False):

    # initialize output matrix
    merged_matrix = np.ones(visual_matrix.shape)
    matrix_size = visual_matrix.shape[0]
    offset = 0 if diagonal_from_second else 1       
    
    # copy below diagonal entries from conceptual matrix
    for row in range(matrix_size):
        for col in range(row + 1):
            merged_matrix[row][col] = conceptual_matrix[row][col]

    # copy above diagonal entries from visual matrix
    for row in range(matrix_size):
        for col in range(row + offset, matrix_size):
            merged_matrix[row][col] = visual_matrix[row][col]
    
    return merged_matrix

# creates a heatmap
def make_heatmap(ax, matrix, legend):
    im = ax.imshow(matrix, cm.get_cmap('gray_r'))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend, fontsize = 8)
    ax.set_yticklabels(legend, fontsize = 8) 

    return im



# load info from pickle files
visual_item_matrix = visual_input_data['similarities']
conceptual_item_matrix = conceptual_input_data['similarities']
item_names = visual_input_data['items']

visual_category_matrix = visual_input_data['category_similarities']
conceptual_category_matrix = conceptual_input_data['category_similarities']
category_names = visual_input_data['categories']

empty_matrix = np.ones(visual_category_matrix.shape)

# create merged matrices
item_matrix = merge_matrices(conceptual_item_matrix, visual_item_matrix)
category_matrix_visual = merge_matrices(empty_matrix, visual_category_matrix, True)
category_matrix_conceptual = merge_matrices(conceptual_category_matrix, empty_matrix)

# set up overall plot for heatmap
fig = plt.figure(figsize = (16,9))
ax_items = fig.add_subplot(121)
ax_cats_vis = fig.add_subplot(222)
ax_cats_con = fig.add_subplot(224)

# add heatmap for item matrix
im = make_heatmap(ax_items, item_matrix, item_names)

# add color bar
cbar = fig.colorbar(im, ticks=[1.1,4.9], ax=ax_items, orientation='vertical')#, pad = 0.4)
cbar.ax.set_ylabel("similarity", rotation='vertical')
cbar.ax.set_yticklabels(['low','high'])

# add heatmaps for category matrices
make_heatmap(ax_cats_vis, category_matrix_visual, category_names)
make_heatmap(ax_cats_con, category_matrix_conceptual, category_names)

# store overall heatmap plot
fig.tight_layout()     
output_file_name = os.path.join(args.output_folder, 'heatmap.png')
fig.savefig(output_file_name, bbox_inches='tight', dpi=200)




# TODO include similarity_correlations.py