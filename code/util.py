# -*- coding: utf-8 -*-
"""
Utility functions used by other scripts

Created on Tue Jun 18 23:09:08 2019

@author: lbechberger
"""

from math import sqrt

# distance functions to use in compute_correlations
distance_functions = {'InnerProduct': {'precompute': lambda x, y: [-1 * x_i * y_i for x_i, y_i in zip(x, y)],
                                        'regression_targets': lambda x: x,
                                        'aggregate': lambda x, w: sum([x_i * w_i * w_i for x_i, w_i in zip(x, w)]),
                                        'weights': lambda x: sqrt(x)}, 
                      'Euclidean': {'precompute': lambda x, y: [(x_i - y_i)*(x_i - y_i) for x_i, y_i in zip(x, y)],
                                    'regression_targets': lambda x: x * x,
                                    'aggregate': lambda x, w: sqrt(sum([x_i * w_i * w_i for x_i, w_i in zip(x, w)])),
                                    'weights': lambda x: sqrt(x)}, 
                      'Manhattan': {'precompute': lambda x, y: [abs(x_i - y_i) for x_i, y_i in zip(x, y)],
                                    'regression_targets': lambda x: x,
                                    'aggregate': lambda x, w: sum([x_i * w_i for x_i, w_i in zip(x, w)]),
                                    'weights': lambda x: x}}

def evaluate_fold(distances, dissimilarities):
    """Evaluates a given fold with predicted distances and target dissimilarities."""

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    # transform input into vectors for correlation computation
    target_vector = np.reshape(dissimilarities, (-1,1)) 
    distance_vector = np.reshape(distances, (-1,1)) 
    
    # compute correlations
    pearson, _ = pearsonr(distance_vector, target_vector)
    spearman, _ = spearmanr(distance_vector, target_vector)
    kendall, _ = kendalltau(distance_vector, target_vector)

    # compute least squares regression for R² metric
    linear_regression = LinearRegression()
    linear_regression.fit(distance_vector, target_vector)
    predictions = linear_regression.predict(distance_vector)
    r2_linear = r2_score(target_vector, predictions)
    
    # compute isotonic regression for R² metric
    x = np.reshape(distance_vector, (-1))
    y = np.reshape(target_vector, (-1))
    isotonic_regression = IsotonicRegression()
    predictions = isotonic_regression.fit_transform(x, y)
    r2_isotonic = r2_score(y, predictions)
    
    return {'pearson': pearson[0], 'spearman': spearman, 'kendall': kendall,
                'r2_linear': r2_linear, 'r2_isotonic': r2_isotonic,
                'targets': target_vector, 'predictions': distance_vector}
   

def precompute_distances(vectors, dissimilarities, distance_function):
    """
    Pre-computes distance information for later correlation analysis, using the given
    vectors and the given matrix of target dissimilarities.
    
    Returns a tuple (precomputed_distances, target_dissimilarities) of two vectors
    which can be used for computing correlations.
    """    

    import numpy as np

    precomputed_distances = []
    target_dissimilarities = []
    # only look at upper triangle of matrix (as it is symmetric and the diagonal is not interesting)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):

            vec_i = np.reshape(vectors[i], -1)
            vec_j = np.reshape(vectors[j], -1)  
            precomputed = distance_functions[distance_function]['precompute'](vec_i, vec_j)
            precomputed_distances.append(precomputed)
            
            target_dissimilarities.append(dissimilarities[i][j])
    
    precomputed_distances = np.array(precomputed_distances)
    target_dissimilarities = np.array(target_dissimilarities)

    return precomputed_distances, target_dissimilarities

def compute_correlations(precomputed_distances, target_dissimilarities, distance_function, n_folds = None, seed = None):
    """
    Computes the correlation between vector distances and actual dissimilarities,
    using the given distance function and the precomputed distances and dissimilarities. 
    If n_folds is not None, then dimension weights are estimated based on a linear regression 
    in an 'n_folds'-fold cross validation (using the given seed to create the folds).
    
    Returns a dictionary from correlation metric to its corresponding value. 
    For convenience, this dictionary also contains both the vector of target dissimilarities
    and the vector of predicted distances as well as the (average) weights for the individual dimensions.
    """  
    
    import numpy as np
    
    if n_folds is None:
        # use fixed weights and evaluate once
        weights = [1]*len(precomputed_distances[0])

        distances = [distance_functions[distance_function]['aggregate'](x, weights) for x in precomputed_distances]
        overall_result = evaluate_fold(distances, target_dissimilarities)
        overall_result['weights'] = weights
        return overall_result
        
    else: # do cross-validation

        from scipy.optimize import nnls
        from sklearn.model_selection import KFold

        # create fold structure        
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = seed)

        weights_list = [] 
        results_list = None

        # iterate through folds        
        for train_index, test_index in kf.split(precomputed_distances):
            
            # collect and prepare training data
            training_distances = precomputed_distances[train_index]
            training_targets = target_dissimilarities[train_index]
            
            # collect and prepare test data
            test_distances = precomputed_distances[test_index]
            test_targets = target_dissimilarities[test_index]
            
            # transform targets if necessary
            transformed_targets = [distance_functions[distance_function]['regression_targets'](x) for x in training_targets]
            transformed_targets = np.array(transformed_targets)

            # do regression
            raw_weights, _ = nnls(training_distances, transformed_targets)

            if sum(raw_weights) == 0:
                # non-negative linear regression did not work --> try with Lasso and very small alpha
                from sklearn.linear_model import Lasso
                lasso = Lasso(alpha = 1e-10, positive = True, random_state=seed)
                lasso.fit(training_distances, transformed_targets)
                raw_weights = lasso.coef_   
                            
            weights = [distance_functions[distance_function]['weights'](w) for w in raw_weights]
            
            if sum(weights) == 0:
                # this should not happen; if it does: warn and skip fold
                print("WARNING! All optimized weights are zero, skipping this fold!")
            else:
                # store weights
                weights_list.append(weights)
                
                # evaluate
                distances = [distance_functions[distance_function]['aggregate'](x, weights) for x in test_distances]
                result = evaluate_fold(distances, test_targets)
                
                # aggregate
                if results_list is None:
                    results_list = {}
                    for key, value in result.items():
                        results_list[key] = [value]
                else:
                    for key, value in result.items():
                        results_list[key].append(value)
    
    
        overall_result = {}
            
        if results_list is not None:
            # average across folds
            for key, value in results_list.items():
                
                if key == 'predictions' or key == 'targets':
                    overall_result[key] = np.concatenate(value, axis = 0)
                else:
                    overall_result[key] = np.mean(value)
            
            overall_result['weights'] = np.mean(weights_list, axis = 0)  
        
        else:
            # worst case: weights have always been zero
            # --> manually set all output values to NaN
            for key in ['pearson', 'spearman', 'kendall', 'r2_linear', 'r2_isotonic', 'weights', 'predictions', 'targets']:
                overall_result[key] = float('NaN')
        
        return overall_result
        


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
        img = block_reduce(array, (block_size, block_size), aggregator_function, cval = 255)
    else:
        array = np.asarray(image.getdata())
        width, height = image.size
        array = np.reshape(array, [width, height, 3])
        img = block_reduce(array, (block_size, block_size, 1), aggregator_function, cval = 255)
    
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
    
    with open(vector_file, 'r') as f_in:
        for line in f_in:
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
    
    category_names = data_set['category_names']
    
    # sort item IDs based on categories
    item_ids = []
    for category in category_names:
        item_ids += data_set['categories'][category]['items']
    
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
        cats = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))
        categories_of_interest = [cat for cat in category_names if cat in cats]
    
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
        cats = list(set(map(lambda x: data_set['items'][x]['category'], items_of_interest)))
        categories_of_interest = [cat for cat in category_names if cat in cats]
        
    elif subset == "cats":
        # consider only the categories from the second study, but use all items within them
        categories_of_interest = ["buildings", "vegetables", "dishes", "insects", "street vehicles", 
                                      "fruits", "electrical appliances", "animals", 
                                      "upper body clothing", "plants", "birds", "tools"]
        items_of_interest = []
        for item in item_ids:
            if data_set['items'][item]['category'] in categories_of_interest:
                items_of_interest.append(item)
    
    # no matter which subset was used: sort the idem IDs and create a corresponding list of item names
    items = list(items_of_interest)
    items_of_interest = []
    for category in categories_of_interest:
        for item in data_set['categories'][category]['items']:
            if item in items:
                items_of_interest.append(item)
    
    return items_of_interest, categories_of_interest


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

def load_item_images(image_folder, item_ids):
    """
    Loads the images for all the given items from the given folder.
    
    Returns a list of images sorted in the same way as item_ids.
    """

    import os    
    from PIL import Image
    
    images = []
    for item in item_ids:
        for file_name in os.listdir(image_folder):
            if os.path.isfile(os.path.join(image_folder, file_name)) and item in file_name:
                # found the corresponding image
                img = Image.open(os.path.join(image_folder, file_name), 'r')
                img = img.convert("RGBA")
                
                # conversion of white to alpha based on https://stackoverflow.com/a/765774
                img_data = img.getdata()
                new_data = []
                for item in img_data:
                    if item[0] == 255 and item[1] == 255 and item[2] == 255:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                img.putdata(new_data)
                images.append(img)
                break
    return images

def create_labeled_scatter_plot(x, y, output_file_name, x_label = "x-axis", y_label = "y-axis", images = None, zoom = 0.15, item_ids = None, directions = None):
    """
    Creates a  labeled scatter plot of the given lists x and y.
    
    Uses the given axis labels and stores the plot under the given file name. 
    If images is a list of PIL Images, they are used instead of points in the scatter plot, scaled with the given zoom.
    If images is None and item_ids is a list of item IDs, they are added as annotation to the points of the scatter plot.
    
    If directions is not None, the directions contained in it are added to the plot.
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np

    fig, ax = plt.subplots(figsize=(12,12))
    if images != None:
        
        # plot scatter plot with images        
        # based on https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
        if ax is None:
            ax = plt.gca()
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0, im0 in zip(x, y, images):
            im = OffsetImage(im0, zoom=zoom)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        ax.scatter(x,y, s=0)

    else:
        # plot scatter plot without images
        ax.scatter(x,y)
        # add item IDs if given
        if item_ids != None:
            for label, x0, y0 in zip(item_ids, x, y):
                plt.annotate(
        		label,
        		xy=(x0, y0), xytext=(-20, 20),
        		textcoords='offset points', ha='right', va='bottom',
        		bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        		arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)

    if directions is not None:
        
        for direction_name, direction_vector in directions.items():
            ax.arrow(0, 0, direction_vector[0], direction_vector[1], head_width = 0.03, color = 'k', zorder=4)
            text_pos_x = direction_vector[0] + np.sign(direction_vector[0])*0.1 
            text_pos_y = direction_vector[1] + np.sign(direction_vector[1])*0.1
            ax.text(text_pos_x, text_pos_y, direction_name, size=16, ha='center', va='center', color='k', zorder=5)
            # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.arrow.html
            # https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/arrow_demo.html#sphx-glr-gallery-text-labels-and-annotations-arrow-demo-py

    fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
    plt.close()


def normalize_direction(v):
    """
    Normalizes the given vector v (i.e., divides it by its length).
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def normalize_vectors(vectors):
    """
    Normalizes a given list of vectors such that their centroid is in the origin 
    and that their root mean squared distance from the origin equals one.
    """

    # first center them
    centroid = np.average(vectors, axis = 0)
    centered_vectors = vectors - centroid
    
    # now rescale them
    squared_distances = np.sum(np.square(centered_vectors), axis = 1)
    mean_squared_distance = np.average(squared_distances)
    root_mean_squared_distance = np.sqrt(mean_squared_distance) 
    normalized_vectors = centered_vectors / root_mean_squared_distance    
    
    return normalized_vectors

def add_correlation_metrics_to_parser(parser):
    """
    Add command line arguments for correlation criteria to the given argument parser.
    """
    parser.add_argument('--pearson', action = 'store_true', help = 'compute and store Pearson correlation')
    parser.add_argument('--spearman', action = 'store_true', help = 'compute and store Spearman correlation')
    parser.add_argument('--kendall', action = 'store_true', help = 'compute and store Kendall correlation')
    parser.add_argument('--r2_linear', action = 'store_true', help = 'compute and store coefficient of determination for linear regression')
    parser.add_argument('--r2_isotonic', action = 'store_true', help = 'compute and store coefficient of determination for isotonic regression')

def get_correlation_metrics_from_args(args):
    """
    Extract a list of correlation metrics to use from the program arguments.
    """
    correlation_metrics = []
    if args.pearson:
        correlation_metrics.append('pearson')
    if args.spearman:
        correlation_metrics.append('spearman')
    if args.kendall:
        correlation_metrics.append('kendall')
    if args.r2_linear:
        correlation_metrics.append('r2_linear')
    if args.r2_isotonic:
        correlation_metrics.append('r2_isotonic')
    return correlation_metrics