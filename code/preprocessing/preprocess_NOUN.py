# -*- coding: utf-8 -*-
"""
Reads the NOUN distance table and stores it in the form of a matrix in a pickle file.

Created on Wed Jan 30 14:15:16 2019

@author: lbechberger
"""

import pickle, argparse, csv
import numpy as np

parser = argparse.ArgumentParser(description='Preprocessing similarity data of the NOUN study')
parser.add_argument('distance_table', help = 'CSV file containing the distance data of the NOUN study')
parser.add_argument('output_file', help = 'path to the output pickle file')
parser.add_argument('-p', '--plot', action="store_true", help = 'plot a histogram of distance values')
args = parser.parse_args()

# read dissimilarity matrix from csv file
with open(args.distance_table, 'r') as f:
    data_iter = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    data = [data for data in data_iter]
    T1 = np.asarray(data)
    # transform items of the array from string to float
    T2 = [[float(letter) for letter in x] for x in T1]
    dissimilarity_matrix = np.asarray(T2)

# create array with names of images "2001" - "2064"
item_names = []
for i in range(2001, 2065):
    item_names.append(str(i))

# put similarity values into dissimilarity matrix
similarity_matrix = np.full((len(item_names), len(item_names)), np.nan)
number_of_filled_entries = 0
constraints_per_item = {}
for item in item_names:
    constraints_per_item[item] = 0

all_similarities = []

largest_entry = np.max(dissimilarity_matrix)
for row in range(64):
    for column in range(64):
        
        # build similarity matrix
        similarity_matrix[row][column] = largest_entry - dissimilarity_matrix[row][column]

        if dissimilarity_matrix[row][column] > 0 or row == column:
            constraints_per_item[item_names[row]] += 1
            number_of_filled_entries += 1
        
        all_similarities.append(similarity_matrix[row][column])

# analyze matrix
matrix_size = len(item_names) * len(item_names)
print("dissimilarity matrix: {0} x {0}, {1} entries, {2} are filled (equals {3}%)".format(len(item_names), 
          matrix_size, number_of_filled_entries, 100*(number_of_filled_entries / matrix_size)))

average_num_constraints = 0
for item, num_constraints in constraints_per_item.items():
    print("{0}: {1} constraints".format(item, num_constraints))
    average_num_constraints += num_constraints
print("average number of constraints per item: {0}".format(average_num_constraints / len(item_names)))

result = {'items': item_names, 'item_names': item_names, 'similarities': similarity_matrix, 'dissimilarities': dissimilarity_matrix}

with open(args.output_file, 'wb') as f:
    pickle.dump(result, f)

# plot the distribution of distances in the distance matrix
if args.plot:
    from matplotlib import pyplot as plt

    output_path = args.output_file.split('.')[0]    
    
    plt.hist(all_similarities, bins=21)
    plt.title('distribution of all similarity values')
    plt.savefig(output_path + '-distr.png', bbox_inches='tight', dpi=200)
    plt.close()

    dissimilarity_values = dissimilarity_matrix.reshape((-1,1))
    plt.hist(dissimilarity_values, bins=21)
    plt.title('distribution of averaged dissimilarity values in matrix')
    plt.savefig(output_path + '-matrix.png', bbox_inches='tight', dpi=200)
    plt.close()          