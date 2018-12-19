# -*- coding: utf-8 -*-
"""
Compute the average image per category

Created on Wed Dec 19 14:57:05 2018

@author: lbechberger
"""

import argparse, pickle, os
from PIL import Image
import numpy as np
from skimage.measure import block_reduce

parser = argparse.ArgumentParser(description='Average category image')
parser.add_argument('input_file', help = 'the input file containing the data set information')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the images should be saved', default='.')
parser.add_argument('-r', '--resolution', type = int, default = 283, help = 'the resolution of the output image in pixels')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
args = parser.parse_args()

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

item_ids = list(data_set['items'].keys())

if args.subset == "all":
    # use all the similarity ratings that we have    
    
    items_of_interest = list(item_ids)

elif args.subset == "between":
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

elif args.subset == "within":
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
    
elif args.subset == "cats":
    # consider only the categories from the second study, but use all items within them
    second_study_categories = ["C03_Elektrogeräte", "C04_Gebäude", "C05_Gemüse", "C06_Geschirr", "C07_Insekten", 
                                   "C10_Landtiere", "C12_Oberkörperbekleidung", "C13_Obst", "C14_Pflanzen", 
                                   "C19_Straßenfahrzeuge", "C21_Vögel", "C25_Werkzeug"]
    items_of_interest = []
    for item in item_ids:
        if data_set['items'][item]['category'] in second_study_categories:
            items_of_interest.append(item)

for category_name, category_dict in data_set['categories'].items():
    category_items = [item for item in category_dict['items'] if item in items_of_interest]
    if len(category_items) == 0:
        continue
    print(category_name, len(category_items))
    
    image = []
    counter = 0
    img_res = 0
    for item_id in category_items:
        for file_name in os.listdir(args.image_folder):
            if os.path.isfile(os.path.join(args.image_folder, file_name)) and item_id in file_name:
                # found the corresponding image: load it and convert to greyscale
                img = Image.open(os.path.join(args.image_folder, file_name), 'r')
                img = img.convert("L")
                
                array = np.asarray(img.getdata())
                array = np.reshape(array, img.size)
                block_size = int(np.ceil(img.size[0] / args.resolution))
                img_res = max(img_res, img.size[0])
                img = block_reduce(array, (block_size, block_size), np.mean)
                
                # make a column vector out of this and store it
                if len(image) == 0:
                    image = img
                else:
                    image += img
                
                counter += 1
                # don't need to look at other files for this item_id, so can break out of inner loop
                break
    image /= counter
    
    # now make the image large again
    pil_img = Image.fromarray(image)
    pil_img = pil_img.convert("L")
    pil_img = pil_img.resize((img_res, img_res), Image.NEAREST)
    pil_img.save(os.path.join(args.output_folder, '{0}.jpg'.format(category_name)))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    