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

for file_name in os.listdir(args.input_folder):
    image = skimage.io.imread(fname=os.path.join(args.input_folder, file_name), as_gray=True)
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    black_and_white = histogram[0] + histogram[255]
    is_binary = black_and_white == sum(histogram)
    print(file_name, is_binary)
    all_binary = all_binary and is_binary

print('\n', all_binary)