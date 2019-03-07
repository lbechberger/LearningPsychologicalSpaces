# -*- coding: utf-8 -*-
"""
Analyze the differences between conceptual similarity and visual similarity.

Created on Thu Mar  7 13:14:23 2019

@author: lbechberger
"""

import pickle, argparse
from scipy.stats import mannwhitneyu
import numpy as np

parser = argparse.ArgumentParser(description='correlation of visual and conceptual similarity ratings')
parser.add_argument('visual_similarity_file', help = 'the input file containing the visual similarity ratings')
parser.add_argument('conceptual_similarity_file', help = 'the input file containing the conceptual similarity ratings')
args = parser.parse_args()

# load the similarity data
with open(args.visual_similarity_file, 'rb') as f_in:
    visual_input_data = pickle.load(f_in)
with open(args.conceptual_similarity_file, 'rb') as f_in:
    conceptual_input_data = pickle.load(f_in)

item_names = visual_input_data['item_names']
visual_dissimilarities = visual_input_data['dissimilarities']
conceptual_dissimilarities = conceptual_input_data['dissimilarities']

# compute the difference between the two matrices and look for item pairs with large and small differences
print("Looking for item pairs with large and small differences")
print("Negative values: conceptually more similar than visually; Positive values: visually more similar than conceptually")
difference_matrix = conceptual_dissimilarities - visual_dissimilarities
upper_threshold = np.percentile(np.abs(difference_matrix), 98)
lower_threshold = np.percentile(np.abs(difference_matrix), 2)

list_of_large_differences = []
list_of_small_differences = []
for row in range(difference_matrix.shape[0]):
    for column in range(row + 1, difference_matrix.shape[1]):
        entry = difference_matrix[row][column]
        item_1 = item_names[row]
        item_2 = item_names[column]
        if abs(entry) >= upper_threshold:
            list_of_large_differences.append((item_1, item_2, entry))
        if abs(entry) <= lower_threshold:
            list_of_small_differences.append((item_1, item_2, entry))

list_of_large_differences = sorted(list_of_large_differences, key = lambda x: abs(x[2]), reverse = True)
list_of_small_differences = sorted(list_of_small_differences, key = lambda x: abs(x[2]))

print('\nlarge differences:')
for item_1, item_2, difference in list_of_large_differences:
    print("\t{0} - {1} : {2}".format(item_1, item_2, difference))

print('\nsmall differences:')
for item_1, item_2, difference in list_of_small_differences:
    print("\t{0} - {1} : {2}".format(item_1, item_2, difference))

# statistical significance test: MannWhitneyU
visual_vector = np.reshape(visual_dissimilarities, (-1))
conceptual_vector = np.reshape(conceptual_dissimilarities, (-1))
mannwhitneyu_stat, mannwhitneyu_p = mannwhitneyu(visual_vector, conceptual_vector)
print("\nStatistical analysis of differences: p = {0} (stat: {1})".format(mannwhitneyu_p, mannwhitneyu_stat))