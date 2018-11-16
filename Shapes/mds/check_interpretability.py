# -*- coding: utf-8 -*-
"""
Checks the interpretability of the space by trying to train linear SVMs.

Created on Fri Nov 16 12:47:16 2018

@author: lbechberger
"""

import argparse, os
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser(description='Finding interpretable directions')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
parser.add_argument('classification_folder', help = 'the folder containing the classification distributions')
args = parser.parse_args()

# read the vectors
vectors = {}
with open(args.vector_file, 'r') as f:
    for line in f:
        vector = []
        tokens = line.replace('\n', '').split(',')
        # first entry is the item ID
        item = tokens[0]
        # all other entries are the coordinates
        vector += list(map(lambda x: float(x), tokens[1:]))
        vectors[item] = vector

for file_name in os.listdir(args.classification_folder):
    positive_items = []
    with open(os.path.join(args.classification_folder, file_name), 'r') as f:
        for line in f:
            positive_items.append(line.replace('\n',''))

    positive_examples = [vectors[item] for item in positive_items]
    negative_examples = [vectors[item] for item in vectors.keys() if item not in positive_items]
    binary_labels = [1]*len(positive_examples) + [0]*len(negative_examples)
    svm = LinearSVC()
    svm.fit(positive_examples + negative_examples, binary_labels)
    direction = svm.coef_
    svm_prediction = svm.predict(positive_examples + negative_examples)
    kappa = cohen_kappa_score(svm_prediction, binary_labels)
    
    print("{0}: kappa {1}, direction {2}".format(file_name, kappa, direction))
    