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

parser = argparse.ArgumentParser(description='correlation of visual and conceptual similarity ratings')
parser.add_argument('visual_similarity_file', help = 'the input file containing the visual similarity ratings')
parser.add_argument('conceptual_similarity_file', help = 'the input file containing the conceptual similarity ratings')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots')
args = parser.parse_args()

# load the similarity data
with open(args.visual_similarity_file, 'rb') as f_in:
    visual_input_data = pickle.load(f_in)
with open(args.conceptual_similarity_file, 'rb') as f_in:
    conceptual_input_data = pickle.load(f_in)

item_ids = visual_input_data['items']
visual_dissimilarities = visual_input_data['dissimilarities']
conceptual_dissimilarities = conceptual_input_data['dissimilarities']

# transform dissimilarity matrices into vectors for correlation computation
visual_vector = np.reshape(visual_dissimilarities, (-1,1)) 
conceptual_vector = np.reshape(conceptual_dissimilarities, (-1,1)) 

# compute correlations
pearson, _ = pearsonr(visual_vector, conceptual_vector)
print("Pearson correlation:", pearson[0])
spearman, _ = spearmanr(visual_vector, conceptual_vector)
print("Spearman correlation:", spearman)
kendall, _ = kendalltau(visual_vector, conceptual_vector)
print("Kendall correlation:", kendall)

# compute least squares regression for R² metric: visual to conceptual
linear_regression = LinearRegression()
linear_regression.fit(visual_vector, conceptual_vector)
predictions = linear_regression.predict(visual_vector)
r2_linear = r2_score(conceptual_vector, predictions)
print("R² linear visual to conceptual:", r2_linear)

# compute least squares regression for R² metric: conceptual to visual
linear_regression = LinearRegression()
linear_regression.fit(conceptual_vector, visual_vector)
predictions = linear_regression.predict(conceptual_vector)
r2_linear = r2_score(visual_vector, predictions)
print("R² linear conceptual to visual:", r2_linear)

# compute isotonic regression for R² metric: visual to conceptual
x = np.reshape(visual_dissimilarities, (-1))
y = np.reshape(conceptual_dissimilarities, (-1))
isotonic_regression = IsotonicRegression()
predictions = isotonic_regression.fit_transform(x, y)
r2_isotonic = r2_score(y, predictions)
print("R² isotonic visual to conceptual:", r2_isotonic)

# compute isotonic regression for R² metric: visual to conceptual
x = np.reshape(conceptual_dissimilarities, (-1))
y = np.reshape(visual_dissimilarities, (-1))
isotonic_regression = IsotonicRegression()
predictions = isotonic_regression.fit_transform(x, y)
r2_isotonic = r2_score(y, predictions)
print("R² isotonic conceptual to visual:", r2_isotonic)
            
if args.plot:
    # create scatter plot if user want us to
    fig, ax = plt.subplots(figsize=(12,12))
    u, c = np.unique(np.c_[visual_vector,conceptual_vector], return_counts=True, axis=0)
    s = lambda x : (((x-x.min())/float(x.max()-x.min())+1)*8)**2
    ax.scatter(u[:,0],u[:,1],s = s(c))
    plt.xlabel('visual dissimilarity')
    plt.ylabel('conceptual dissimilarity')
    plt.title('scatter plot of visual and conceptual dissimilarity')

    output_file_name = os.path.join(args.output_folder, 'scatter.png')        
    
    fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
    plt.close()