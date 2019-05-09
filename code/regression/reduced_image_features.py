# -*- coding: utf-8 -*-
"""
Creates feature vectors for each image by reducing them to a lower resolution.

Created on Thu May  9 08:06:21 2019

@author: lbechberger
"""

import pickle, argparse, os
from PIL import Image
import numpy as np
from skimage.measure import block_reduce
import tensorflow as tf

parser = argparse.ArgumentParser(description='Feature Extraction based on reducing the image size')
parser.add_argument('input_dir', help = 'input folder containing the augmented data')
parser.add_argument('output_file', help = 'output file for the feature vectors')
parser.add_argument('-g', '--greyscale', action = 'store_true', help = 'only consider greyscale information (i.e., collapse color channels)')
parser.add_argument('-a', '--aggregator', default = 'mean', help = 'aggregator function to use when downscaling the images')
parser.add_argument('-b', '--block_size', type = int, default = 1, help = 'block size to use when downscaling the images')
args = parser.parse_args()

aggregator_functions = {'max': np.max, 'mean': np.mean, 'min': np.min, 'std': np.std, 'var': np.var, 'median': np.median, 'product': np.prod}

def downscale_image(image, aggregator, block_size, greyscale):
    
    if greyscale:
        img = image.convert("L")
        array = np.asarray(img.getdata())
        array = np.reshape(array, img.size)
        img = block_reduce(array, (block_size, block_size), aggregator_functions[aggregator])
    else:
        array = np.asarray(image.getdata())
        width, height = image.size
        array = np.reshape(array, [width, height, 3])
        img = block_reduce(array, (block_size, block_size, 3), aggregator_functions[aggregator])
    # make a column vector out of this and store it
    return np.reshape(img, (1,-1))

# need to convert tensorflow string representation into numbers
tf_image_string = tf.placeholder(tf.string)
decoder = tf.image.decode_jpeg(tf_image_string)

result = {}
image_file_names = [f for f in os.listdir(args.input_dir)]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for image_file in image_file_names:
        image_name = image_file.split('.')[0]
        print("processing {0}".format(image_name))
        images = pickle.load(open(os.path.join(args.input_dir, image_file), 'rb'))
        
        downscaled_images = []
        for image in images:
            pixels = session.run(decoder, feed_dict = {tf_image_string : image})
            img = Image.fromarray(pixels, 'RGB')
            downscaled = downscale_image(img, args.aggregator, args.block_size, args.greyscale)
            downscaled_images.append(downscaled)

        result[image_name] = downscaled_images

pickle.dump(result, open(args.output_file, 'wb'))