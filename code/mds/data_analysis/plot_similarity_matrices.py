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
def merge_matrices(visual_matrix, conceptual_matrix, double_diagonal):

    # initialize output matrix
    output_rows, output_columns = visual_matrix.shape
    offset = 3 if double_diagonal else 0 # two extra columns if diagonal needs to be duplicated
    output_columns += offset
    merged_matrix = np.ones((output_rows, output_columns))
    
    # copy below diagonal entries from conceptual matrix
    for row in range(output_rows):
        for col in range(row + 1):
            merged_matrix[row][col] = conceptual_matrix[row][col]

    # copy above diagonal entries from visual matrix
    for row in range(output_rows):
        for col in range(row, output_columns - offset):
            merged_matrix[row][col + offset] = visual_matrix[row][col]
    
    return merged_matrix

# creates a heatmap
def make_heatmap(ax, matrix, legend, double_diagonal):
    im = ax.imshow(matrix, cm.get_cmap('gray_r'))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    if double_diagonal:
        # TODO: take care of additional labels
        pass

    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend) 
    
    return im

# does the actual plotting
def make_heatmaps(visual_matrix, conceptual_matrix, legend, output_file_name, mode='merge'):

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
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(unified_matrix, cm.get_cmap('gray_r'))
    
    if mode == 'visual':
        ax.tick_params(bottom=False,left=False,top=True,right=True, labeltop=True, labelright=True, labelbottom=False, labelleft=False)    
        plt.setp(ax.get_xticklabels(), rotation='vertical')
    else:
        plt.setp(ax.get_xticklabels(), rotation='vertical')
        cbar = ax.figure.colorbar(im, ticks=[1.1,4.9], ax=ax, orientation='vertical')
        cbar.ax.set_ylabel("similarity", rotation='vertical')
        cbar.ax.set_yticklabels(['low','high'])


    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend)
    ax.set_yticklabels(legend) 
 

    fig.tight_layout()     
    fig.savefig(output_file_name, bbox_inches='tight', dpi=200)


# load info from pickle files
visual_item_matrix = visual_input_data['similarities']
conceptual_item_matrix = conceptual_input_data['similarities']
item_names = visual_input_data['items']

visual_category_matrix = visual_input_data['category_similarities']
conceptual_category_matrix = conceptual_input_data['category_similarities']
category_names = visual_input_data['categories']

# create merged matrices
item_matrix = merge_matrices(visual_item_matrix, conceptual_item_matrix, False)
category_matrix = merge_matrices(visual_category_matrix, conceptual_category_matrix, True)

# set up overall plot for heatmap
fig = plt.figure(figsize = (16,9))
ax_items = fig.add_subplot(121)
ax_colorbar = fig.add_subplot(222)
ax_categories = fig.add_subplot(224)

# add heatmap for item matrix
im = make_heatmap(ax_items, item_matrix, item_names, False)

# add heatmap for category matrix
make_heatmap(ax_categories, category_matrix, category_names, True)

# TODO: make color bar narrower and 
# add color bar
cbar = ax_colorbar.figure.colorbar(im, ticks=[1.1,4.9], cax=ax_colorbar, orientation='horizontal')
cbar.ax.set_xlabel("similarity")#, rotation='vertical')
cbar.ax.set_xticklabels(['low','high'])
# store overall heatmap plot
fig.tight_layout()     
output_file_name = os.path.join(args.output_folder, 'heatmap.png')
fig.savefig(output_file_name, bbox_inches='tight', dpi=200)


#make_heatmaps(visual_item_matrix, conceptual_item_matrix, visual_input_data['item_names'], file_name)
#
## now do category-based analysis
#
## do the plotting
#visual_file_name = os.path.join(args.output_folder, 'heatmap_category_based_visual.png')
#conceptual_file_name = os.path.join(args.output_folder, 'heatmap_category_based_conceptual.png')
#make_heatmaps(visual_category_matrix, conceptual_category_matrix, category_names, visual_file_name, mode='visual')
#make_heatmaps(visual_category_matrix, conceptual_category_matrix, category_names, conceptual_file_name, mode='conceptual')


# TODO include similarity_correlations.py