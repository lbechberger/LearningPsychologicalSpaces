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
parser.add_argument('-o', '--output_size', type = int, help = 'size of output image', default = 224)
parser.add_argument('-m', '--minimum_size', type = int, help = 'minimal size of output object', default = 168)
args = parser.parse_args()

pixel_histogram = [0]*256
possible_sizes = list(range(args.minimum_size, args.output_size + 1))
size_histogram = [0]*(len(possible_sizes))

if args.seed is not None:
    np.random.seed(args.seed)

with open(args.folds_file, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        image_path = row['path']
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]
        fold = row['fold']

        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # find bounding box
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        x_low = args.output_size - 1
        y_low = args.output_size - 1
        x_high = 0
        y_high = 0
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            x_low = min(x_low, x)
            y_low = min(y_low, y)
            x_high = max(x_high, x + w)
            y_high = max(y_high, y + h)

        cropped = image[y_low:y_high, x_low:x_high]
        w = x_high - x_low
        h = y_high - y_low

        # compute sizes
        cropped_size = max(h,w)
        cropped_aspect_ratio = min(h,w)/max(h,w)

        size_combinations = [(args.output_size - i + 1)*int(args.output_size - i*cropped_aspect_ratio + 1) for i in possible_sizes]
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
            x_offset = np.random.randint(args.output_size + 1 - dim[0])
            y_offset = np.random.randint(args.output_size + 1 - dim[1])
            augmented_translations.append((x_offset, y_offset))

        # create images
        augmented_images = []
        
        for dim, translation in zip(augmented_dims, augmented_translations):

            rescaled = cv2.resize(cropped, dim)

            top = translation[1]
            bottom = (args.output_size - translation[1] - dim[1])
            left = translation[0]
            right = (args.output_size - translation[0] - dim[0])
            
            padded = cv2.copyMakeBorder(rescaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

            augmented_images.append(padded)

        # export images as png files
        for idx, img in enumerate(augmented_images):
            output_path = os.path.join(args.output_directory, fold, "{0}-{1}.png".format(image_name, idx))
            cv2.imwrite(output_path, img)

        # store histogram and size information
        for s in augmented_sizes:
            size_histogram[s - args.minimum_size] += 1
        
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
    print('{0},{1}'.format(args.minimum_size + idx, val))

print('\nPIXEL')
for idx, val in enumerate(pixel_histogram):
    print('{0},{1}'.format(idx, val))

# store labels


