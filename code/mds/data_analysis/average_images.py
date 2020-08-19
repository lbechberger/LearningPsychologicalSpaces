# -*- coding: utf-8 -*-
"""
Compute the average image per category

Created on Wed Dec 19 14:57:05 2018

@author: lbechberger
"""

import argparse, pickle, os
from PIL import Image
import numpy as np
from code.util import aggregator_functions, load_image_files_pixel, downscale_image

parser = argparse.ArgumentParser(description='Average category image')
parser.add_argument('input_file', help = 'the input file containing the data set information')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the images should be saved', default='.')
parser.add_argument('-r', '--resolution', type = int, default = 283, help = 'the resolution of the output image in pixels')
parser.add_argument('-a', '--aggregator', help = 'aggregator to use when downscaling the images', default = 'mean')
args = parser.parse_args()

# load the data set from the pickle file
with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

item_names = list(data_set['items'].keys())

images = load_image_files_pixel(item_names, args.image_folder)

for category_name, category_dict in data_set['categories'].items():
        
    image = []
    counter = 0
    img_res = 0
    for item in category_dict['items']:
        
        img_idx = item_names.index(item)
        img = images[img_idx]
        img_res = max(img_res, img.size[0])
        block_size = int(np.ceil(img.size[0] / args.resolution))
        reduced_resolution = int(np.ceil(img_res / block_size))
        small_img, size = downscale_image(img, aggregator_functions[args.aggregator], block_size, True, (reduced_resolution, reduced_resolution))
        
        # pad the image with zeroes if necessary
        if reduced_resolution != args.resolution:
            pad_size = (args.resolution - reduced_resolution) / 2
            pad_before = int(pad_size)
            pad_after = int(np.ceil(pad_size))
            small_img = np.pad(small_img, (pad_before, pad_after), mode = 'constant', constant_values = 255)
            
            print("Padded image from {0} to {1} pixels: Before {2}, after {3}".format(reduced_resolution, args.resolution, pad_before, pad_after))
        
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    