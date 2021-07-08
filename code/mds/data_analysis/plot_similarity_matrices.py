# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:37:53 2020

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from scipy.stats import kendalltau

parser = argparse.ArgumentParser(description='Plotting similarity tables')
parser.add_argument('first_similarity_file', help = 'the input file containing the first set of similarity ratings')
parser.add_argument('second_similarity_file', help = 'the input file containing the second set of similarity ratings')
parser.add_argument('output_folder', help = 'the folder where the plots will be stored')
parser.add_argument('-f', '--first_name', help = 'name of the first set of similarity ratings', default = 'first study')
parser.add_argument('-s', '--second_name', help = 'name of the second set of similarity ratings', default = 'second study')
parser.add_argument('-d', '--dissimilarities', action = 'store_true', help = 'use dissimilarities instead of similarities for the scatter plot')
args = parser.parse_args()

# merge the two given matrices
def merge_matrices(first_matrix, second_matrix, diagonal_from_second = False):

    # initialize output matrix
    merged_matrix = np.ones(second_matrix.shape)
    matrix_size = second_matrix.shape[0]
    offset = 0 if diagonal_from_second else 1       
    
    # copy below diagonal entries from first matrix
    for row in range(matrix_size):
        for col in range(row + 1):
            merged_matrix[row][col] = first_matrix[row][col]

    # copy above diagonal entries from second matrix
    for row in range(matrix_size):
        for col in range(row + offset, matrix_size):
            merged_matrix[row][col] = second_matrix[row][col]
    
    return merged_matrix

# creates a heatmap
def make_heatmap(ax, matrix, legend, fontsize):
    im = ax.imshow(matrix, cm.get_cmap('gray_r'))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    ax.set_xticks(np.arange(len(legend)))
    ax.set_yticks(np.arange(len(legend)))
    ax.set_xticklabels(legend, fontsize = fontsize)
    ax.set_yticklabels(legend, fontsize = fontsize) 

    return im

def make_colorbar(fig, ax, im, fontsize):
    cbar = fig.colorbar(im, ticks=[1.1,4.9], ax=ax, orientation='vertical')
    cbar.ax.set_ylabel("similarity", rotation='vertical', fontsize = fontsize + 2)
    cbar.ax.set_yticklabels(['low','high'], fontsize = fontsize)



# load the similarity data
with open(args.first_similarity_file, 'rb') as f_in:
    first_input_data = pickle.load(f_in)
with open(args.second_similarity_file, 'rb') as f_in:
    second_input_data = pickle.load(f_in)

# load info from pickle files
first_item_matrix = first_input_data['similarities']
second_item_matrix = second_input_data['similarities']
item_names = second_input_data['items']

first_category_matrix = first_input_data['category_similarities']
second_category_matrix = second_input_data['category_similarities']
category_names = second_input_data['categories']

empty_matrix = np.ones(second_category_matrix.shape)

# create merged matrices
item_matrix = merge_matrices(first_item_matrix, second_item_matrix)
category_matrix_first = merge_matrices(first_category_matrix, empty_matrix)
category_matrix_second = merge_matrices(empty_matrix, second_category_matrix, True)



# set up overall plot for item heatmap
fig_items = plt.figure(figsize = (12,12))
ax_items = fig_items.add_subplot(111)

# add heatmap for item matrix
im = make_heatmap(ax_items, item_matrix, item_names, 12)

# add color bar
make_colorbar(fig_items, ax_items, im, 12)

# store item heatmap plot
fig_items.tight_layout()     
output_file_name_heatmap_items = os.path.join(args.output_folder, 'heatmap_{0}_{1}_items.png'.format(args.first_name, args.second_name))
fig_items.savefig(output_file_name_heatmap_items, bbox_inches='tight', dpi=200)



# set up overall plot for category heatmap
fig_cats = plt.figure(figsize = (18,9))
ax_cats_second = fig_cats.add_subplot(121)
ax_cats_first = fig_cats.add_subplot(122)

# add heatmaps for category matrices
make_heatmap(ax_cats_first, category_matrix_first, category_names, 14)
make_heatmap(ax_cats_second, category_matrix_second, category_names, 14)

ax_cats_second.set_xlabel("(a)", fontsize = 24)
ax_cats_first.set_xlabel("(b)", fontsize = 24)

# add color bar
fig_cats.subplots_adjust(right=0.85)
ax_cats_cbar = fig_cats.add_axes([0.8, 0.15, 0.1, 0.7])
ax_cats_cbar.axis('off')
make_colorbar(fig_cats, ax_cats_cbar, im, 14)

# store category heatmap plot
output_file_name_heatmap_cats = os.path.join(args.output_folder, 'heatmap_{0}_{1}_categories.png'.format(args.first_name, args.second_name))
fig_cats.savefig(output_file_name_heatmap_cats, bbox_inches='tight', dpi=200)


# replace item matrix with dissimilarities if necessary!
if args.dissimilarities:
    first_item_matrix = first_input_data['dissimilarities']
    second_item_matrix = second_input_data['dissimilarities']

# transform item-based similarity matrices into vectors for scatter plot
first_vector = np.reshape(first_item_matrix, (-1,1)) 
second_vector = np.reshape(second_item_matrix, (-1,1)) 

# prepare sizes of individual data points
fig, ax = plt.subplots(figsize=(12,12))
u, c = np.unique(np.c_[first_vector,second_vector], return_counts=True, axis=0)
s = lambda x : ((8*(x-x.min())/float(x.max()-x.min())+1)*8)**2

# plot and label
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.scatter(u[:,0],u[:,1],s = s(c))

if args.dissimilarities:
    plt.xlabel('{0} Dissimilarity'.format(args.first_name), fontsize = 20)
    plt.ylabel('{0} Dissimilarity'.format(args.second_name), fontsize = 20)
    plt.title('Scatter Plot of {0} and {1} Dissimilarity'.format(args.first_name, args.second_name), fontsize = 20)
else:
    plt.xlabel('{0} Similarity'.format(args.first_name), fontsize = 20)
    plt.ylabel('{0} Similarity'.format(args.second_name), fontsize = 20)
    plt.title('Scatter Plot of {0} and {1} Similarity'.format(args.first_name, args.second_name), fontsize = 20)

# store the overall scatter plot
output_file_name_scatter = os.path.join(args.output_folder, 'scatter_{0}_{1}.png'.format(args.first_name, args.second_name))        
fig.savefig(output_file_name_scatter, bbox_inches='tight', dpi=200)

# compute correlation coefficient
kendall, _ = kendalltau(first_vector, second_vector)
print("rank correlation (Kendall's tau): ", kendall)