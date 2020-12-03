# -*- coding: utf-8 -*-
"""
Create histograms of Shape images.

Created on Thu Dec  3 10:36:18 2020

@author: lbechberger
"""

import argparse, os
import skimage.io
import numpy as np

parser = argparse.ArgumentParser(description='Scan Shape images')
parser.add_argument('input_folder', help = 'folder containing the Shape images')
args = parser.parse_args()

all_binary = True
fill_sizes = []

for file_name in os.listdir(args.input_folder):
    image = skimage.io.imread(fname=os.path.join(args.input_folder, file_name), as_gray=True)
    histogram, bin_edges = np.histogram(image, bins=256)
    black_and_white = histogram[0] + histogram[255]
    is_binary = black_and_white == sum(histogram)

    min_x = 282
    min_y = 282
    max_x = 0
    max_y = 0

    for x in range(283):
        for y in range(283):
            img_val = image[x][y] if np.isscalar(image[x][y]) else image[x][y][0]
            if img_val < 64:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
#            elif img_val < 255:
#                print(img_val)
    
    width = max_x - min_x
    height = max_y - min_y
    if width < 0:
        print(width, max_x, min_x)
    if height < 0:
        print(height, max_y, min_y)
    
    fill_size = max(width,height)/283
    
    print(file_name, is_binary, fill_size)
    all_binary = all_binary and is_binary
    fill_sizes.append(fill_size)

print('\n', all_binary, np.mean(fill_sizes), np.std(fill_sizes), np.min(fill_sizes), np.max(fill_sizes))