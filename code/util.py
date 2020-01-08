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
    For convenience, this dictionary also contains both the vector of target dissimilarities
    and the vector of predicted similarities.
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
                'r2_linear': r2_linear, 'r2_isotonic': r2_isotonic,
                'targets': target_vector, 'predictions': sim_vector}


def extract_inception_features(images, model_dir, output_shape):
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
            inception_features.append(feature_vector.reshape(output_shape))
                
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


def downscale_images(images, aggregator_function, block_size, greyscale, output_shape):
    """
    Downscales all of the given images via block_reduce using the given aggregator and block size.
    """
    
    result = []
    for image in images:
        downscaled_image, _ = downscale_image(image, aggregator_function, block_size, greyscale, output_shape)
        result.append(downscaled_image)  
    return result

def load_image_files_pixel(item_ids, image_folder):
    """
    Loads all image files in the format needed for the pixel baseline.
    """
    import os
    from PIL import Image

    images = []
    for item_id in item_ids:
        for file_name in os.listdir(image_folder):
            if os.path.isfile(os.path.join(image_folder, file_name)) and item_id in file_name:
                # found the corresponding image: load it
                img = Image.open(os.path.join(image_folder, file_name), 'r')
                images.append(img)
                
                # don't need to look at other files for this item_id, so can break out of inner loop
                break

    return images

def load_image_files_ann(item_ids, image_folder):
    """
    Loads all image files in the format needed for the ANN baseline.
    """

    import os
    from tensorflow.python.platform import gfile

    images = []
    for item_id in item_ids:
        for file_name in os.listdir(image_folder):
            if os.path.isfile(os.path.join(image_folder, file_name)) and item_id in file_name:
                # found the corresponding image: load it
                image_data = gfile.FastGFile(os.path.join(image_folder, file_name), 'rb').read()            
                images.append(image_data)

                # don't need to look at other files for this item_id, so can break out of inner loop
                break

    return images

def load_mds_vectors(vector_file, item_ids = None):
    """
    Loads the MDS vectors from the given file. 
    
    Returns either a dictionary from item ID to vector or a list of vectors (ordered by item ID),
    depending on whether a list of item IDs is given.
    """

    result_dict = {}
    result_list = []    
    
    with open(vector_file, 'r') as f:
        for line in f:
            tokens = line.replace('\n','').split(',')
            item = tokens[0]
            vector = list(map(lambda x: float(x), tokens[1:]))
            
            result_dict[item] = vector
    
    if item_ids is not None:
        for item_id in item_ids:
            result_list.append(np.reshape(result_dict[item_id], (1,-1)))
        return result_list
        
    return result_dict


def select_data_subset(subset, data_set):
    """
    Select a subset of the given data set.
    
    The parameter 'subset' can have the following values: 'all', 'between', 'within', 'cats'.
    Returns a triple of lists: items_of_interest, item_names, and categories_of_interest.
    """
    
    item_ids = list(data_set['items'].keys())
    category_names = list(data_set['categories'].keys())
    
    if subset == "all":
        # use all the similarity ratings that we have    
        
        items_of_interest = list(item_ids)
        categories_of_interest = list(category_names)
    
    elif subset == "between":
        # only use the similarity ratings from the 'between' file
    
        items_of_interest = []   
        
        for idx1, item1 in enumerate(item_ids):
            for idx2, item2 in enumerate(item_ids):
                
                if idx2 <= idx1:
                    continue
                
                tuple_id = str(sorted([item1, item2]))
                if tuple_id in data_set['similarities']:
                    border = data_set['similarities'][tuple_id]['border']
                    between_ratings = data_set['similarities'][tuple_id]['values'][border:]
                    if len(between_ratings) > 0:
                        items_of_interest.append(item1)
                        items_of_interest.append(item2)
        
        items_of_interest = list(set(items_of_interest)) # remove duplicates
        categories_of_interest = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))
    
    elif subset == "within":
        # only use the similarity ratings from the 'within' file
        items_of_interest = []   
        
        for idx1, item1 in enumerate(item_ids):
            for idx2, item2 in enumerate(item_ids):
                
                if idx2 <= idx1:
                    continue
                
                tuple_id = str(sorted([item1, item2]))
                if tuple_id in data_set['similarities']:
                    border = data_set['similarities'][tuple_id]['border']
                    between_ratings = data_set['similarities'][tuple_id]['values'][:border]
                    if len(between_ratings) > 0:
                        items_of_interest.append(item1)
                        items_of_interest.append(item2)
        
        items_of_interest = list(set(items_of_interest)) # remove duplicates
        categories_of_interest = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))
        
    elif subset == "cats":
        # consider only the categories from the second study, but use all items within them
        categories_of_interest = ["C03_Elektrogeräte", "C04_Gebäude", "C05_Gemüse", "C06_Geschirr", "C07_Insekten", 
                                       "C10_Landtiere", "C12_Oberkörperbekleidung", "C13_Obst", "C14_Pflanzen", 
                                       "C19_Straßenfahrzeuge", "C21_Vögel", "C25_Werkzeug"]
        items_of_interest = []
        for item in item_ids:
            if data_set['items'][item]['category'] in categories_of_interest:
                items_of_interest.append(item)
    
    # no matter which subset was used: sort the idem IDs and create a corresponding list of item names
    items_of_interest = sorted(items_of_interest)
    item_names = list(map(lambda x: data_set['items'][x]['name'], items_of_interest))
    categories_of_interest = sorted(categories_of_interest)
    
    return items_of_interest, item_names, categories_of_interest


def find_limit(subset, data_set, items_of_interest):
    """
    Find the minimum number of similarity ratings to use as a limit.
    Look only at the given subset of the given data_set concerning the given set of items_of_interest.
    """    
    
    limit = 1000
    for idx1, item1 in enumerate(items_of_interest):
        for idx2, item2 in enumerate(items_of_interest):
            if idx2 <= idx1:
                continue
            
            tuple_id = str(sorted([item1, item2]))
            if tuple_id in data_set['similarities']:
                similarity_ratings = data_set['similarities'][tuple_id]['values']
                if subset == "between":
                    # remove everything from first study
                    border = data_set['similarities'][tuple_id]['border']
                    similarity_ratings = similarity_ratings[border:]
                elif subset == "within":
                    # remove everything from second study
                    border = data_set['similarities'][tuple_id]['border']
                    similarity_ratings = similarity_ratings[:border]
                
                if len(similarity_ratings) > 0:
                    # only adapt the limit if there are any ratings left
                    limit = min(limit, len(similarity_ratings))
    return limit