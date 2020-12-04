# -*- coding: utf-8 -*-
"""
Create histograms of images.

Created on Thu Dec  3 10:36:18 2020

@author: lbechberger
"""

import argparse, os, pickle
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Scan images')
parser.add_argument('input_folder', help = 'folder containing the Shape images')
parser.add_argument('image_size', type = int, help = 'width and height of the image')
parser.add_argument('output_file', help = 'path to output pickle file')
args = parser.parse_args()

all_binary = True
fill_sizes = []
all_pixels = []
all_images = []

for file_name in os.listdir(args.input_folder):
    image = skimage.io.imread(fname=os.path.join(args.input_folder, file_name), as_gray=True)
    histogram, bin_edges = np.histogram(image, bins=256)
    black_and_white = histogram[0] + histogram[255]
    is_binary = black_and_white == sum(histogram)

    min_x = args.image_size - 1
    min_y = args.image_size - 1
    max_x = 0
    max_y = 0

    simple_img = np.zeros((args.image_size, args.image_size))

    for x in range(args.image_size):
        for y in range(args.image_size):
            img_val = image[x][y] if np.isscalar(image[x][y]) else image[x][y][0]
            if img_val < 64:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            all_pixels.append(img_val)
            simple_img[x][y] = img_val
    
    width = max_x - min_x
    height = max_y - min_y
    if width < 0:
        print(width, max_x, min_x)
    if height < 0:
        print(height, max_y, min_y)
    
    fill_size = max(width,height)/args.image_size
    
    print(file_name, is_binary, fill_size)
    all_binary = all_binary and is_binary
    fill_sizes.append(fill_size)
    
    all_images.append(simple_img)

print('\n', all_binary, np.mean(fill_sizes), np.std(fill_sizes), np.min(fill_sizes), np.max(fill_sizes))
n, _, _ = plt.hist(all_pixels, bins=256, log=True)
plt.show()
print('\n'.join(map(lambda x: str(x), n)))

with open(args.output_file, 'wb') as f:
    pickle.dump(all_images, f)