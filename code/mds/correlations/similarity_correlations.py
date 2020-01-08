# -*- coding: utf-8 -*-
"""
Computing correlations between visual similarity and conceptual similarity.

Created on Thu Mar  7 12:47:52 2019

@author: lbechberger
"""

import pickle, argparse, os
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, kendalltau
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='correlation of similarity ratings from different studies')
parser.add_argument('first_similarity_file', help = 'the input file containing the first set of similarity ratings')
parser.add_argument('second_similarity_file', help = 'the input file containing the second set of similarity ratings')
parser.add_argument('-f', '--first_name', help = 'name of the first set of similarity ratings', default = 'first study')
parser.add_argument('-s', '--second_name', help = 'name of the second set of similarity ratings', default = 'second study')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots')
args = parser.parse_args()

# load the similarity data
with open(args.first_similarity_file, 'rb') as f_in:
    first_input_data = pickle.load(f_in)
with open(args.second_similarity_file, 'rb') as f_in:
    second_input_data = pickle.load(f_in)

item_ids = first_input_data['items']
first_dissimilarities = first_input_data['dissimilarities']
second_dissimilarities = second_input_data['dissimilarities']

# transform dissimilarity matrices into vectors for correlation computation
first_vector = np.reshape(first_dissimilarities, (-1,1)) 
second_vector = np.reshape(second_dissimilarities, (-1,1)) 

# compute correlations
pearson, _ = pearsonr(first_vector, second_vector)
print("Pearson correlation:", pearson[0])
spearman, _ = spearmanr(first_vector, second_vector)
print("Spearman correlation:", spearman)
kendall, _ = kendalltau(first_vector, second_vector)
print("Kendall correlation:", kendall)

# compute least squares regression for R² metric: first to second
linear_regression = LinearRegression()
linear_regression.fit(first_vector, second_vector)
predictions = linear_regression.predict(first_vector)
r2_linear = r2_score(second_vector, predictions)
print("R² linear {0} to {1}:".format(args.first_name, args.second_name), r2_linear)

# compute least squares regression for R² metric: second to first
linear_regression = LinearRegression()
linear_regression.fit(second_vector, first_vector)
predictions = linear_regression.predict(second_vector)
r2_linear = r2_score(first_vector, predictions)
print("R² linear {0} to {1}:".format(args.second_name, args.first_name), r2_linear)

# compute isotonic regression for R² metric: first to second
x = np.reshape(first_dissimilarities, (-1))
y = np.reshape(second_dissimilarities, (-1))
isotonic_regression = IsotonicRegression()
predictions = isotonic_regression.fit_transform(x, y)
r2_isotonic = r2_score(y, predictions)
print("R² isotonic {0} to {1}:".format(args.first_name, args.second_name), r2_isotonic)

# compute isotonic regression for R² metric: visual to conceptual
x = np.reshape(second_dissimilarities, (-1))
y = np.reshape(first_dissimilarities, (-1))
isotonic_regression = IsotonicRegression()
predictions = isotonic_regression.fit_transform(x, y)
r2_isotonic = r2_score(y, predictions)
print("R² isotonic {0} to {1}:".format(args.second_name, args.first_name), r2_isotonic)
            
if args.plot:
    # create scatter plot if user want us to
    fig, ax = plt.subplots(figsize=(12,12))
    u, c = np.unique(np.c_[first_vector,second_vector], return_counts=True, axis=0)
    s = lambda x : (((x-x.min())/float(x.max()-x.min())+1)*8)**2
    ax.scatter(u[:,0],u[:,1],s = s(c))
    plt.xlabel('{0} dissimilarity'.format(args.first_name))
    plt.ylabel('{0} dissimilarity'.format(args.second_name))
    plt.title('scatter plot of {0} and {1} dissimilarity'.format(args.first_name, args.second_name))

    output_file_name = os.path.join(args.output_folder, 'scatter.png')        
    
    fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
    plt.close()