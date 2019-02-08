# -*- coding: utf-8 -*-
"""
Computes the correlation of the distances from the MDS space to the human similarity ratings.

Created on Thu Dec  6 11:43:04 2018

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, kendalltau
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='correlation of MDS distances to human similarity ratings')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('mds_folder', help = 'the folder containing the MDS vectors')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('--n_min', type = int, default = 1, help = 'the smallest space to investigate')
parser.add_argument('--n_max', type = int, default = 20, help = 'the largest space to investigate')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots of distances vs. dissimilarities')
args = parser.parse_args()

def cosine(x,y):
    return cosine_distances(x,y)[0][0]

def euclidean(x,y):
    return euclidean_distances(x,y)[0][0]

def manhattan(x,y):
    return manhattan_distances(x,y)[0][0]

scoring_functions = {'Cosine': cosine, 'Euclidean': euclidean, 'Manhattan': manhattan}

# set up file name for output file
_, path_and_file = os.path.splitdrive(args.similarity_file)
_, file = os.path.split(path_and_file)
file_without_extension = file.split('.')[0]
output_file_name = os.path.join(args.output_folder, "{0}-MDS.csv".format(file_without_extension))

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

with open(output_file_name, 'w', buffering=1) as f_out:

    f_out.write("n_dims,scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")

    for number_of_dimensions in range(args.n_min, args.n_max + 1):
        
        n_dim_vecs = {}
        vectors = []  
        # load the vectors -- first into a dictionary ...
        with open(os.path.join(args.mds_folder, "{0}D-vectors.csv".format(number_of_dimensions)), 'r') as f_in:
            for line in f_in:
                tokens = line.replace('\n','').split(',')
                item = tokens[0]
                vector = list(map(lambda x: float(x), tokens[1:]))
                n_dim_vecs[item] = np.array(vector)
        
        # ... then in correct ordering into a list      
        for item_id in item_ids:
            vectors.append(np.reshape(n_dim_vecs[item_id], (1,-1)))
        
        for scoring_name, scoring_function in scoring_functions.items():
            
            # build a matrix of similarity scores
            dissimilarity_scores = np.ones(target_dissimilarities.shape)
            
            for i in range(len(item_ids)):
                for j in range(len(item_ids)):
                    sim = scoring_function(vectors[i], vectors[j])
                    dissimilarity_scores[i][j] = sim
            
            # transform dissimilarity matrices into vectors for correlation computation
            target_vector = np.reshape(target_dissimilarities, (-1,1)) 
            sim_vector = np.reshape(dissimilarity_scores, (-1,1)) 
            
            # compute correlations
            pearson, _ = pearsonr(sim_vector, target_vector)
            spearman, _ = spearmanr(sim_vector, target_vector)
            kendall, _ = kendalltau(sim_vector, target_vector)
        
            # compute least squares regression for R² metric
            linear_regression = LinearRegression()
            linear_regression.fit(sim_vector, target_vector)
            predictions = linear_regression.predict(sim_vector)
            r2_linear = r2_score(target_vector, predictions)
            
            # compute isotonic regression for R² metric
            x = np.reshape(dissimilarity_scores, (-1))
            y = np.reshape(target_dissimilarities, (-1))
            isotonic_regression = IsotonicRegression()
            predictions = isotonic_regression.fit_transform(x, y)
            r2_isotonic = r2_score(y, predictions)
            
            # compute and store correlation
            f_out.write("{0},{1},{2},{3},{4},{5},{6}\n".format(number_of_dimensions, scoring_name, pearson[0], spearman, 
                                                                kendall, r2_linear, r2_isotonic))
            
            if args.plot:
                # create scatter plot if user want us to
                fig, ax = plt.subplots(figsize=(12,12))
                ax.scatter(sim_vector,target_vector)
                plt.xlabel('MDS distance')
                plt.ylabel('real distance')
                plt.title('scatter plot {0}D {1} distance'.format(number_of_dimensions, scoring_name))
        
                output_file_name = os.path.join(args.output_folder, '{0}D-{1}.png'.format(number_of_dimensions, scoring_name))        
                
                fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
                plt.close()