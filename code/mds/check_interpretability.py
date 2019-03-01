# -*- coding: utf-8 -*-
"""
Checks the interpretability of the space by trying to train linear SVMs.

Created on Fri Nov 16 12:47:16 2018

@author: lbechberger
"""

import argparse, os
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score
import numpy as np

parser = argparse.ArgumentParser(description='Finding interpretable directions')
parser.add_argument('vector_file', help = 'the input file containing the vectors')
parser.add_argument('classification_folder', help = 'the folder containing the classification distributions')
parser.add_argument('n', type = int, help = 'the number of dimensions in the underlying space')
parser.add_argument('-o', '--output_file', help = 'output csv file for collecting the results', default = None)
parser.add_argument('-r', '--repetitions', type = int, help = 'number of repetitions in sampling the baselines', default = 20)
parser.add_argument('-b', '--baseline', action = "store_true", help = 'whether or not to compute the random baselines')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation when computing baselines', default = None)
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

kappas = {'MDS':[], 'uniform':[], 'normal':[], 'shuffled':[]}
category_names = []

if args.seed is not None:
    np.random.seed(args.seed)

print("INDIVIDUAL")

for file_name in os.listdir(args.classification_folder):
    positive_items = []
    with open(os.path.join(args.classification_folder, file_name), 'r') as f:
        for line in f:
            positive_items.append(line.replace('\n',''))

    positive_examples = [vectors[item] for item in positive_items]
    negative_examples = [vectors[item] for item in vectors.keys() if item not in positive_items]
    all_examples = positive_examples + negative_examples
    binary_labels = [1]*len(positive_examples) + [0]*len(negative_examples)

    svm = LinearSVC()
    svm.fit(all_examples, binary_labels)
    direction = svm.coef_
    kappa = cohen_kappa_score(svm.predict(all_examples), binary_labels)
    
    kappa_uniform = 0
    kappa_normal = 0
    kappa_shuffled = 0    
    
    if args.baseline:
        
        for i in range(args.repetitions):
            uniform_points = np.random.rand(len(all_examples), args.n)
            uniform_svm = LinearSVC()
            uniform_svm.fit(uniform_points, binary_labels)
            kappa_uniform += cohen_kappa_score(uniform_svm.predict(uniform_points), binary_labels)       
            
            normal_points = np.random.normal(size=(len(all_examples), args.n))
            normal_svm = LinearSVC()
            normal_svm.fit(normal_points, binary_labels)
            kappa_normal += cohen_kappa_score(normal_svm.predict(normal_points), binary_labels)       
            
            shuffled_points = np.array(list(all_examples))
            np.random.shuffle(shuffled_points)
            shuffled_svm = LinearSVC()
            shuffled_svm.fit(shuffled_points, binary_labels)
            kappa_shuffled += cohen_kappa_score(shuffled_svm.predict(shuffled_points), binary_labels)       
        
        kappa_uniform /= args.repetitions
        kappa_normal /= args.repetitions
        kappa_shuffled /= args.repetitions
    
    kappas['MDS'].append(kappa)
    kappas['uniform'].append(kappa_uniform)
    kappas['normal'].append(kappa_normal)
    kappas['shuffled'].append(kappa_shuffled)   
    
    category_names.append(file_name)
    
    print("\t{0}: kappa {1}, direction {2} (uniform: {3}, normal: {4}, shuffled: {5})".format(file_name, 
              kappa, direction, kappa_uniform, kappa_normal, kappa_shuffled))

def avg(x):
    return sum(x)/len(x)
print("\nOVERALL")
print("\tmin: {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(min(kappas['MDS']), min(kappas['uniform']),
                                                                      min(kappas['normal']), min(kappas['shuffled'])))
print("\tavg: {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(avg(kappas['MDS']), avg(kappas['uniform']),
                                                                      avg(kappas['normal']), avg(kappas['shuffled'])))
print("\tmax: {0} (uniform: {1}, normal: {2}, shuffled: {3})".format(max(kappas['MDS']), max(kappas['uniform']),
                                                                      max(kappas['normal']), max(kappas['shuffled'])))
          
if args.output_file != None:
    
    # write headline if necessary
    if not os.path.exists(args.output_file):

        # construct headline
        headline = 'dims'
        for category in category_names:
            headline += ',{0},{0}_u,{0}_n,{0}_s'.format(category)
        
        with open(args.output_file, 'w') as f:
            f.write(headline)
            f.write("\n")
    
    # write content
    with open(args.output_file, 'a') as f:
        line_items = [args.n]
        for mds, uniform, normal, shuffled in zip(kappas['MDS'], kappas['uniform'], kappas['normal'], kappas['shuffled']):
            line_items += [mds, uniform, normal, shuffled]
        f.write(",".join(map(lambda x: str(x), line_items)))
        f.write("\n")