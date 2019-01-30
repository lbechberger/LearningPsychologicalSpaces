# -*- coding: utf-8 -*-
"""
Computes the different similarity measures  between the images on a pixel basis.
Compares these similarities to the human similarity ratings.

Created on Tue Dec  4 09:27:06 2018

@author: lbechberger
"""

import pickle, argparse, os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from skimage.measure import block_reduce
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Pixel-based similarity baseline')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='analysis')
parser.add_argument('-s', '--size', type = int, default = 283, help = 'the size of the image, used to determine the maximal block size')
args = parser.parse_args()

aggregator_functions = {'max': np.max, 'mean': np.mean, 'min': np.min, 'std': np.std, 'var': np.var, 'median': np.median, 'product': np.prod}
scoring_functions = {'Cosine': cosine_distances, 'Euclidean': euclidean_distances, 'Manhattan': manhattan_distances}

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
            # found the corresponding image: load it and convert to greyscale
            img = Image.open(os.path.join(args.image_folder, file_name), 'r')
            img = img.convert("L")
            images.append(img)
            
            # don't need to look at other files for this item_id, so can break out of inner loop
            break

with open(output_file_name, 'w', buffering=1) as f:

    f.write("aggregator,block_size,image_size,scoring,pearson,spearman,kendall,r2\n")
    last_image_size = 9999
    for block_size in range(1, args.size + 1):
        
        # if the resulting image size is the same: skip this block size (will just introduce more noise due to zero padding)
        current_image_size = int(np.ceil(args.size / block_size))
        if current_image_size == last_image_size:
            continue
        else:
            last_image_size = current_image_size
            print(current_image_size)

        for aggregator_name, aggregator_function in aggregator_functions.items():
        
            if block_size == 1 and aggregator_name in ['std', 'var', 'mean', 'min', 'median', 'product']:
                # can't compute std or var on a single element
                # value for all others is identical to max, so only compute once
                continue
            
            transformed_images = []
            for img in images:
                # transform image via block_reduce
                array = np.asarray(img.getdata())
                array = np.reshape(array, img.size)
                img = block_reduce(array, (block_size, block_size), aggregator_function)
                
                image_size = img.shape[0]
                # make a column vector out of this and store it
                img = np.reshape(img, (1,-1))
                transformed_images.append(img)
    
            for scoring_name, scoring_function in scoring_functions.items():
                dissimilarity_scores = np.ones(target_dissimilarities.shape)
                
                for i in range(len(item_ids)):
                    for j in range(len(item_ids)):
    
                        img_i = transformed_images[i]
                        img_j = transformed_images[j]    
                        score = scoring_function(img_i, img_j)[0][0]
                        dissimilarity_scores[i][j] = score
                
                # transform dissimilarity matrices into vectors for correlation computation
                target_vector = np.reshape(target_dissimilarities, (-1,1)) 
                sim_vector = np.reshape(dissimilarity_scores, (-1,1)) 
                pearson, _ = pearsonr(sim_vector, target_vector)
                spearman, _ = spearmanr(sim_vector, target_vector)
                kendall, _ = kendalltau(sim_vector, target_vector)
            
                # compute least squares regression for R^2 metric
                y = np.reshape(target_dissimilarities, (-1))
                x = np.reshape(dissimilarity_scores, (-1))
                A = np.vstack([x, np.ones(len(sim_vector))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]         
                predictions = m*x + c
                r2 = r2_score(y,predictions)
                
                f.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(aggregator_name, block_size, image_size, scoring_name, 
                                                                    abs(pearson[0]), abs(spearman), abs(kendall), r2))
                
                
                # temporary: display scatter plot
                fig, ax = plt.subplots(figsize=(12,12))
                ax.scatter(sim_vector,target_vector)
                plt.xlabel('image distance')
                plt.ylabel('real distance')
                plt.title('scatter plot {0} {1} {2} distance'.format(block_size, aggregator_name, scoring_name))
        
                output_file_name = os.path.join(args.output_folder, '{0}-{1}-{2}.png'.format(block_size, aggregator_name, scoring_name))        
                
                fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
                plt.close()