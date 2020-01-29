# -*- coding: utf-8 -*-
"""
Compute the average image per category

Created on Wed Dec 19 14:57:05 2018

@author: lbechberger
"""

import argparse, pickle, os
from PIL import Image
import numpy as np
from code.util import aggregator_functions, select_data_subset, load_image_files_pixel, downscale_image

parser = argparse.ArgumentParser(description='Average category image')
parser.add_argument('input_file', help = 'the input file containing the data set information')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the images should be saved', default='.')
parser.add_argument('-r', '--resolution', type = int, default = 283, help = 'the resolution of the output image in pixels')
parser.add_argument('-s', '--subset', help = 'the subset of data to use', default="all")
parser.add_argument('-a', '--aggregator', help = 'aggregator to use when downscaling the images', default = 'mean')
args = parser.parse_args()

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

items_of_interest, item_names, categories_of_interest = select_data_subset(args.subset, data_set)
images = load_image_files_pixel(items_of_interest, args.image_folder)

for category_name, category_dict in data_set['categories'].items():
    category_items = [item for item in category_dict['items'] if item in items_of_interest]
    if len(category_items) == 0:
        continue
    print(category_name, len(category_items))
    
    image = []
    counter = 0
    img_res = 0
    for item_id in category_items:
        
        img_idx = items_of_interest.index(item_id)
        img = images[img_idx]
        img_res = max(img_res, img.size[0])
        block_size = int(np.ceil(img.size[0] / args.resolution))
        small_img, size = downscale_image(img, aggregator_functions[args.aggregator], block_size, True, (args.resolution, args.resolution))
                
        # add to the average image of this category
        if len(image) == 0:
            image = small_img
        else:
            image += small_img
        
        counter += 1

    image = np.array(image, dtype=float)
    image /= counter
    
    # now make the image large again
    pil_img = Image.fromarray(image)
    pil_img = pil_img.convert("L")
    pil_img = pil_img.resize((img_res, img_res), Image.NEAREST)
    pil_img.save(os.path.join(args.output_folder, '{0}.jpg'.format(category_name)))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    