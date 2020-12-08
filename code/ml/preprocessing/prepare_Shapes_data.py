# -*- coding: utf-8 -*-
"""
Prepares the data set for our Shapes study.

Read in images, augment them by scaling and translating, and store them as png files.
Create pickle files with noise for regression baseline.
Store labels if applicable.

Bounding box detection based on https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python/21108680
and https://stackoverflow.com/questions/13887863/extract-bounding-box-and-save-it-as-an-image

Created on Thu Dec  3 10:36:18 2020

@author: lbechberger
"""

import argparse, os, pickle, csv
#import skimage.io
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='Prepare the data set for the Shapes study')
parser.add_argument('folds_file', help = 'csv file containing fold information')
parser.add_argument('output_directory', help = 'path to output directory')
parser.add_argument('factor', type = int, help = 'number of augmented samples per original image')
parser.add_argument('-p', '--pickle_output', help = 'prefix of pickle output files (if desired)', default = None)
parser.add_argument('-n', '--noise_prob', nargs = '+', type = float, help = 'noise levels to use for pickle output', default = None)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

pixel_histogram = [0]*256
possible_sizes = list(range(168,225))
size_histogram = [0]*57

image_counter = 0

if args.seed is not None:
    np.random.seed(args.seed)

with open(args.folds_file, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        image_path = row['path']
        fold = row['fold']

        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # find bounding box
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        largest = (0,0,0,0)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > largest[2]*largest[3]:
                largest = (x,y,w,h)

        cropped = image[y:y+h, x:x+w]

        # compute sizes
        cropped_size = max(h,w)
        cropped_aspect_ratio = min(h,w)/max(h,w)

        size_combinations = [(225-i)*int(225-i*cropped_aspect_ratio) for i in possible_sizes]
        size_probabilities = [i/sum(size_combinations) for i in size_combinations]
        
        augmented_sizes = np.random.choice(possible_sizes, args.factor, p=size_probabilities)

        augmented_dims = []
        for size in augmented_sizes:
            if w > h:
                dims = (size, int(size*cropped_aspect_ratio))
            else:
                dims = (int(size*cropped_aspect_ratio), size)
            augmented_dims.append(dims)

        # compute translations
        augmented_translations = []
        for dim in augmented_dims:
            x_offset = np.random.randint(225-dim[0])
            y_offset = np.random.randint(225-dim[1])
            augmented_translations.append((x_offset, y_offset))

        # create images
        augmented_images = []
        
        for dim, translation in zip(augmented_dims, augmented_translations):

            rescaled = cv2.resize(cropped, dim)

            top = translation[1]
            bottom = (224 - translation[1] - dim[1])
            left = translation[0]
            right = (224 - translation[0] - dim[0])
            
            padded = cv2.copyMakeBorder(rescaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

            augmented_images.append(padded)

        # export images as png files
        for img in augmented_images:
            output_path = os.path.join(args.output_directory, fold, "{0}.png".format(image_counter))
            image_counter += 1
            cv2.imwrite(output_path, img)

        # store histogram and size information
        for s in augmented_sizes:
            size_histogram[s-168] += 1
        
        for img in augmented_images:
            img_hist, _ = np.histogram(img, bins=256)
            for i in range(256):
                pixel_histogram[i] += img_hist[i]


        if args.pickle_output is not None:

            if args.noise_prob is None:
                raise(Exception("Need noise probability for pickle export!"))

            for noise_prob in args.noise_prob:
                pass
            
                # TODO apply salt and pepper

                # TODO store pickle output

# print overall histograms of sizes and pixel values
print('SIZE')
for idx, val in enumerate(size_histogram):
    print('{0},{1}'.format(168+idx,val))

print('\nPIXEL')
for idx, val in enumerate(pixel_histogram):
    print('{0},{1}'.format(idx,val))

# store labels


