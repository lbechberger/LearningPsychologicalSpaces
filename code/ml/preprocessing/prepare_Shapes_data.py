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
import numpy as np
import cv2
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from code.util import salt_and_pepper_noise


parser = argparse.ArgumentParser(description='Prepare the data set for the Shapes study')
parser.add_argument('folds_file', help = 'csv file containing fold information')
parser.add_argument('output_directory', help = 'path to output directory')
parser.add_argument('factor', type = int, help = 'number of augmented samples per original image')
parser.add_argument('-p', '--pickle_output_folder', help = 'folder for pickle output files (if desired)', default = None)
parser.add_argument('-n', '--noise_prob', nargs = '+', type = float, help = 'noise levels to use for pickle output', default = None)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-o', '--output_size', type = int, help = 'size of output image', default = 128)
parser.add_argument('-m', '--minimum_size', type = int, help = 'minimal size of output object', default = 96)
parser.add_argument('-f', '--flip_probability', type = float, help = 'probability of horizontal flips', default = 0.0)
parser.add_argument('-r', '--rotation_angle', type = int, help = 'maximal rotation angle', default = 0)
parser.add_argument('-a', '--shear_angle', type = int, help = 'maximal shear angle', default = 0)
args = parser.parse_args()

pixel_histogram = [0]*256
possible_sizes = list(range(args.minimum_size, args.output_size + 1))
size_histogram = [0]*(len(possible_sizes))

pickle_output = {}

if args.seed is not None:
    np.random.seed(args.seed)
    ia.seed(args.seed)

if args.pickle_output_folder is not None:
    folds_info = []

with open(args.folds_file, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        image_path = row['path']
        image_name = os.path.splitext(os.path.split(image_path)[1])[0]
        fold = row['fold']
        if fold not in pickle_output:
            pickle_output[fold] = []
        
        additional_info = None
        if 'class' in row:
            additional_info = row['class']
        elif 'id' in row:
            additional_info = row['id']

        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply augmentation if applicable (flip/rotation/shear)
        seq = iaa.Sequential([iaa.Fliplr(args.flip_probability),
                              iaa.Affine(rotate = (-args.rotation_angle, args.rotation_angle),
                                         mode = "constant", cval = 255),
                              iaa.Affine(shear = (-args.shear_angle, args.shear_angle),
                                         mode = "constant", cval = 255)],
                              random_order = True)
        
        base_images = []
        for i in range(args.factor):
            base_images.append(seq.augment_image(image))

        augmented_images = []
        for base_image in base_images:

            # find bounding box
            thresh = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
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
    
            cropped = base_image[y_low:y_high, x_low:x_high]
            w = x_high - x_low
            h = y_high - y_low

            # compute size
            cropped_size = max(h,w)
            cropped_aspect_ratio = min(h,w)/max(h,w)
    
            size_combinations = [(args.output_size - i + 1)*int(args.output_size - i*cropped_aspect_ratio + 1) for i in possible_sizes]
            size_probabilities = [i/sum(size_combinations) for i in size_combinations]
            
            augmented_size = np.random.choice(possible_sizes, p=size_probabilities)
            size_histogram[augmented_size - args.minimum_size] += 1

            if w > h:
                dims = (augmented_size, int(augmented_size*cropped_aspect_ratio))
            else:
                dims = (int(augmented_size*cropped_aspect_ratio), augmented_size)
    
            # compute translation
            x_offset = np.random.randint(args.output_size + 1 - dims[0])
            y_offset = np.random.randint(args.output_size + 1 - dims[1])
            
            # create image
            rescaled = cv2.resize(cropped, dims)

            top = y_offset
            bottom = (args.output_size - y_offset - dims[1])
            left = x_offset
            right = (args.output_size - x_offset - dims[0])
            
            padded = cv2.copyMakeBorder(rescaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

            augmented_images.append(padded)

        # export images as png files
        for idx, img in enumerate(augmented_images):
            output_path = os.path.join(args.output_directory, fold, "{0}-{1}.png".format(image_name, idx))
            cv2.imwrite(output_path, img)
            
            pickle_output[fold].append((output_path, additional_info))

        # store histogram information        
        for img in augmented_images:
            img_hist, _ = np.histogram(img, bins=256)
            for i in range(256):
                pixel_histogram[i] += img_hist[i]


        if args.pickle_output_folder is not None:

            if args.noise_prob is None:
                raise(Exception("Need noise probability for pickle export!"))

            # store folds info for later
            folds_info.append('{0},{1}\n'.format(additional_info, fold))

            # define tensorflow graph: image --> string
            tf_image = tf.placeholder(tf.uint8, shape=(args.output_size, args.output_size, 1))
            encoder = tf.image.encode_jpeg(tf_image)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
            
                for noise_prob in args.noise_prob:
                
                    # apply salt and pepper noise
                    corrupted_images = []
                    for img in augmented_images:
                        noisy_img = salt_and_pepper_noise(img, noise_prob, args.output_size, 255)
                        encoded_image = session.run(encoder, feed_dict = {tf_image : noisy_img})
                        corrupted_images.append(encoded_image)


                    # store pickle output
                    noise_output_path = os.path.join(args.pickle_output_folder, str(noise_prob), "{0}.pickle".format(additional_info))
                    with open(noise_output_path, 'wb') as f_out:
                        pickle.dump(corrupted_images, f_out)

if args.pickle_output_folder is not None:            
    # need different format for folds information --> create manually
    folds_output_path = os.path.join(args.pickle_output_folder, "folds.csv")
    with open(folds_output_path, 'w') as f_out:
        for line in folds_info:
            f_out.write(line)

# print overall histograms of sizes and pixel values
print('SIZE')
for idx, val in enumerate(size_histogram):
    print('{0},{1}'.format(args.minimum_size + idx, val))

print('\nPIXEL')
for idx, val in enumerate(pixel_histogram):
    print('{0},{1}'.format(idx, val))

# store paths and labels (in shuffled order!)
pickle_output_path = os.path.join(args.output_directory, "{0}.pickle".format(os.path.splitext(os.path.split(args.folds_file)[1])[0]))
for fold in pickle_output.keys():
    np.random.shuffle(pickle_output[fold])
    
with open(pickle_output_path, 'wb') as f_out:
    pickle.dump(pickle_output, f_out)

