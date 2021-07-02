# -*- coding: utf-8 -*-
"""
Uses the feature vectors of the inception network to compute similarity ratings between images.
Checks how correlated they are to the original human similarity ratings.

Created on Sun May 12 07:56:40 2019

@author: lbechberger
"""

import pickle, argparse
from code.util import precompute_distances, compute_correlations, distance_functions
from code.util import extract_inception_features
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args

parser = argparse.ArgumentParser(description='ANN-based similarity baseline')
parser.add_argument('model_dir', help = 'folder for storing the pretrained network')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('distance_file', help = 'the pickle file containing the pre-computed distances')
parser.add_argument('output_file', help = 'the csv file to which the output should be saved')
parser.add_argument('-i', '--image_folder', help = 'the folder containing the original images', default = None)
parser.add_argument('-n', '--n_folds', type = int, help = 'number of folds to use for cross-validation when optimizing weights', default = 5)
parser.add_argument('-s', '--seed', type = int, help = 'fixed seed to use for creating the folds', default = None)
add_correlation_metrics_to_parser(parser)
args = parser.parse_args()

correlation_metrics = get_correlation_metrics_from_args(args)

def load_image_files_ann(item_ids, image_folder):
    """
    Loads all image files in the format needed for the ANN baseline.
    """

    import os
    from tensorflow.python.platform import gfile

    images = []

    for item in item_ids:
        candidates = []
        for file_name in os.listdir(image_folder):
            if os.path.isfile(os.path.join(image_folder, file_name)) and item in file_name:
                candidates.append(file_name)
        if len(candidates) > 1:
            # special case handling for 'owl'/'bowl' and similar
            filtered = [file_name for file_name in candidates if (('_' + item) in file_name and (item + '.') in file_name)]
            
            if len(filtered) == 1:
                file_name = filtered[0]
            else:
                file_name = candidates[0]
        elif len(candidates) == 1:
            file_name = candidates[0]
        else:
            continue

        # found the corresponding image: load it
        image_data = gfile.FastGFile(os.path.join(image_folder, file_name), 'rb').read()            
        images.append(image_data)

    return images

# load the real similarity data
with open(args.similarity_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

items = input_data['items']
target_dissimilarities = input_data['dissimilarities']

if args.image_folder is not None:
    # load images and extract feature vectors
    images = load_image_files_ann(items, args.image_folder) 
    inception_features = extract_inception_features(images, args.model_dir, (1, -1))
    distances = {}
else:
    # load pre-computed distances
    with open(args.distance_file, 'rb') as f_in:
        distances = pickle.load(f_in)

print('extracted features')

with open(args.output_file, 'w', buffering=1) as f_out:

    f_out.write("scoring,weights,{0}\n".format(','.join(correlation_metrics)))
           
    for distance_function in sorted(distance_functions.keys()):
        
        if args.image_folder is not None:
            # precompute distances and targets based on the ann features
            precomputed_distances = precompute_distances(inception_features, distance_function)
            distances[distance_function] = precomputed_distances
        else:
            # simply grab them from the loaded dictionary
            precomputed_distances = distances[distance_function]
        
        # raw correlation
        correlation_results = compute_correlations(precomputed_distances, target_dissimilarities, distance_function)
        f_out.write("{0},fixed,{1}\n".format(distance_function, ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))

        # correlation with optimized weights
        correlation_results = compute_correlations(precomputed_distances, target_dissimilarities, distance_function, args.n_folds, args.seed)
        f_out.write("{0},optimized,{1}\n".format(distance_function, ','.join(map(lambda x: str(correlation_results[x]), correlation_metrics))))

        print('done with {0}; weights: {1}'.format(distance_function, correlation_results['weights']))

# output the collected distances if necessary
if args.image_folder is not None:
    with open(args.distance_file, 'wb') as f_out:
        pickle.dump(distances, f_out)