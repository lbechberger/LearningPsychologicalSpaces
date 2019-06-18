# -*- coding: utf-8 -*-
"""
Uses the feature vectors of the inception network to compute similarity ratings between images.
Checks how correlated they are to the original human similarity ratings.

Created on Sun May 12 07:56:40 2019

@author: lbechberger
"""

import pickle, argparse, os
from tensorflow.python.platform import gfile
from code.util import compute_correlations, distance_functions, extract_inception_features

parser = argparse.ArgumentParser(description='Pixel-based similarity baseline')
parser.add_argument('model_dir', help = 'folder for storing the pretrained network')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots of distances vs. dissimilarities')
args = parser.parse_args()

# set up file name for output file
_, path_and_file = os.path.splitdrive(args.similarity_file)
_, file = os.path.split(path_and_file)
file_without_extension = file.split('.')[0]
output_file_name = os.path.join(args.output_folder, "{0}.csv".format(file_without_extension))

# load the real similarity data
with open(args.similarity_file, 'rb') as f:
    input_data = pickle.load(f)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

# load all images
images = []
for item_id in item_ids:
    for file_name in os.listdir(args.image_folder):
        if os.path.isfile(os.path.join(args.image_folder, file_name)) and item_id in file_name:
            # found the corresponding image: load it
            image_data = gfile.FastGFile(os.path.join(args.image_folder, file_name), 'rb').read()            
            images.append(image_data)
 
inception_features = extract_inception_features(images, args.model_dir)

print('extracted features')

with open(output_file_name, 'w', buffering=1) as f:

    f.write("scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")
           
    for distance_name, distance_function in distance_functions.items():
        
        correlation_metrics = compute_correlations(inception_features, target_dissimilarities, distance_function)
        f.write("{0},{1},{2},{3},{4},{5}\n".format(distance_name, 
                                                    correlation_metrics['pearson'], 
                                                    correlation_metrics['spearman'], 
                                                    correlation_metrics['kendall'], 
                                                    correlation_metrics['r2_linear'], 
                                                    correlation_metrics['r2_isotonic']))

        print('done with {0}'.format(distance_name))
        
#        if args.plot:
#            # create scatter plot if user want us to
#            fig, ax = plt.subplots(figsize=(12,12))
#            ax.scatter(sim_vector,target_vector)
#            plt.xlabel('ANN-based Distance', fontsize = 20)
#            plt.ylabel('Dissimilarity from Psychological Study', fontsize = 20)
#                    
#            output_file_name = os.path.join(args.output_folder, '{0}.png'.format(scoring_name))        
#            
#            fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
#            plt.close()