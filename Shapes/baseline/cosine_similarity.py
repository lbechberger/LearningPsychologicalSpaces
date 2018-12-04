# -*- coding: utf-8 -*-
"""
Computes the cosine similarity between the images on a pixel basis.

Created on Tue Dec  4 09:27:06 2018

@author: lbechberger
"""

import pickle, argparse, os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='Pixel-based cosine similarity baseline')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('image_folder', help = 'the folder containing the original images')
args = parser.parse_args()

with open(args.similarity_file, 'rb') as f:
    input_data = pickle.load(f)

item_ids = input_data['items']
target_similarities = input_data['similarities']

# load all images
images = []
for item_id in item_ids:
    for file_name in os.listdir(args.image_folder):
        if os.path.isfile(os.path.join(args.image_folder, file_name)) and item_id in file_name:
            # found the corresponding image
            img = Image.open(os.path.join(args.image_folder, file_name), 'r')
            img = img.convert("L")

            # TODO: convert to 5x5 image here            
            
            img_vector = np.asarray(img.getdata())
            img = np.reshape(img_vector, (1,-1)) # make a column vector out of this

            images.append(img)
            break

cosine_similarities = np.ones(target_similarities.shape)

for i in range(len(item_ids)):
    for j in range(len(item_ids)):
        cos_sim = cosine_similarity(images[i], images[j])
        cosine_similarities[i][j] = cos_sim[0][0]

target_vector = np.reshape(target_similarities, (-1)) 
cosine_vector = np.reshape(cosine_similarities, (-1)) 

correlation = pearsonr(cosine_vector, target_vector)
print(correlation[0])