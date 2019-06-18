# -*- coding: utf-8 -*-
"""
Utility functions used by other scripts

Created on Tue Jun 18 23:09:08 2019

@author: lbechberger
"""

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

# distance functions to use in compute_correlations
distance_functions = {'Cosine': cosine_distances, 'Euclidean': euclidean_distances, 'Manhattan': manhattan_distances}


def compute_correlations(vectors, dissimilarities, distance_function):
    """
    Computes the correlation between vector distances and actual dissimilarities,
    using the given distance function between the vectors.
    
    Returns a dictionary from correlation metric to its corresponding value.
    """    
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr, spearmanr, kendalltau

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


def extract_inception_features(images, model_dir):
    """
    Loads the inception network and extracts features for all the given items.
    
    Uses the provided model_dir to store the ANN, searches for all item_ids in the image_folder.
    Returns a list of feature vectors, ordered in the same way as item_ids.
    Based on https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11
    """
    import os, sys, tarfile
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    from six.moves import urllib
   
    # download the inception network
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(model_dir)
    
    # load the computation graph
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
    # finally extract the features        
    inception_features = []
    with tf.Session() as sess:
        second_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            
        for image in images:
            feature_vector = sess.run(second_to_last_tensor, {'DecodeJpeg/contents:0': image})
            inception_features.append(feature_vector.reshape(1, -1))
                
    return inception_features

import numpy as np
# aggregator functions for block_reduce
aggregator_functions = {'max': np.max, 'mean': np.mean, 'min': np.min, 'median': np.median}

def downscale_image(image, aggregator_function, block_size, greyscale, output_shape):
    """
    Downscales the given image via block_reduce using the given aggregator and block size.
    """
    import numpy as np
    from skimage.measure import block_reduce

    if greyscale:
        img = image.convert("L")
        array = np.asarray(img.getdata())
        array = np.reshape(array, img.size)
        img = block_reduce(array, (block_size, block_size), aggregator_function)
    else:
        array = np.asarray(image.getdata())
        width, height = image.size
        array = np.reshape(array, [width, height, 3])
        img = block_reduce(array, (block_size, block_size, 1), aggregator_function)
    
    image_size = img.shape[0]
    # make a column vector out of this and store it
    result = np.reshape(img, output_shape)
    return result, image_size


