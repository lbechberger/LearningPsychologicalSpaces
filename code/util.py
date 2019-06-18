# -*- coding: utf-8 -*-
"""
Utility functions used by other scripts

Created on Tue Jun 18 23:09:08 2019

@author: lbechberger
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

# distance functions to use in compute_correlations
distance_functions = {'Cosine': cosine_distances, 'Euclidean': euclidean_distances, 'Manhattan': manhattan_distances}


def compute_correlations(vectors, dissimilarities, distance_function):
    """
    Computes the correlation between vector distances and actual dissimilarities,
    using the given distance function between the vectors.
    
    Returns a dictionary from correlation metric to its corresponding value.
    """    
    
    # initialize dissimilarities with ones (arbitrary, will be overwritten anyways)
    dissimilarity_scores = np.ones(dissimilarities.shape)
                
    for i in range(len(vectors)):
        for j in range(len(vectors)):

            vec_i = vectors[i]
            vec_j = vectors[j]    
            score = distance_function(vec_i, vec_j)[0][0]
            dissimilarity_scores[i][j] = score
                
    # transform dissimilarity matrices into vectors for correlation computation
    target_vector = np.reshape(dissimilarities, (-1,1)) 
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
    y = np.reshape(dissimilarities, (-1))
    isotonic_regression = IsotonicRegression()
    predictions = isotonic_regression.fit_transform(x, y)
    r2_isotonic = r2_score(y, predictions)
    
    return {'pearson': pearson[0], 'spearman': spearman, 'kendall': kendall,
                'r2_linear': r2_linear, 'r2_isotonic': r2_isotonic}