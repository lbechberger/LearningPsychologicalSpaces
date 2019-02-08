# -*- coding: utf-8 -*-
"""
Runs MDS on the given data set with the given configuration.

Created on Mon Oct 22 14:44:04 2018

@author: lbechberger
"""

import pickle, argparse
import matplotlib.pyplot as plt
from sklearn import manifold
import os

parser = argparse.ArgumentParser(description='MDS for shapes')
parser.add_argument('input_file', help = 'the input file to use')
parser.add_argument('-n', '--n_init', type = int, help = 'number random starts', default = 4)
parser.add_argument('-d', '--dims', type = int, help = 'highest number of dimensions to check', default = 20)
parser.add_argument('-i', '--max_iter', type = int, help = 'maximum number of iterations', default = 300)
parser.add_argument('-e', '--export', help = 'path for export', default = None)
parser.add_argument('-p', '--plot', action = 'store_true', help = 'Plot the stress curve')
parser.add_argument('-m', '--metric', action = 'store_true', help = 'Compute metric version of MDS')
parser.add_argument('-s', '--seed', type = int, help = 'seed for the random number generator', default = None)
args = parser.parse_args()

with open(args.input_file, 'rb') as f:
    input_data = pickle.load(f)

items_of_interest = input_data['items']
dissimilarity_matrix = input_data['dissimilarities']

# run MDS
plot_coordinates = []
for number_of_dimensions in range(1, args.dims + 1):
    mds = manifold.MDS(n_components=number_of_dimensions, dissimilarity="precomputed", metric=args.metric, 
                       n_init = args.n_init, max_iter = args.max_iter, random_state = args.seed, n_jobs = -1)
    results = mds.fit(dissimilarity_matrix)
    
    if args.export != None:
        # need to export
        with open(os.path.join(args.export, "{0}D-vectors.csv".format(number_of_dimensions)), 'w') as f:
            for index, item in enumerate(items_of_interest):
                f.write("{0},{1}\n".format(item, ','.join(map(lambda x: str(x), results.embedding_[index]))))
    
    plot_coordinates.append([number_of_dimensions, results.stress_])
    print("{0},{1}".format(number_of_dimensions, results.stress_))

if args.plot:
    plt.plot(list(map(lambda x: x[0], plot_coordinates)), list(map(lambda x: x[1], plot_coordinates)), marker='o', linestyle='dashed')
    plt.xlabel('number of dimensions')
    plt.ylabel('stress')
    plt.title('Stress of {0} MDS (SMACOF)'.format('metric' if args.metric else 'nonmetric'))
    plt.savefig(os.path.join(args.export, "Scree.png"), bbox_inches='tight', dpi=200)
    