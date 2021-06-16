# -*- coding: utf-8 -*-
"""
Visualize the reconstructions made by a given model.

Created on Fri Jun  4 13:31:11 2021

@author: lbechberger
"""

import argparse
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from code.ml.ann.keras_utils import SaltAndPepper
from code.util import salt_and_pepper_noise

parser = argparse.ArgumentParser(description='Visualizing reconstructions')
parser.add_argument('network_file', help = 'hdf5 file containing the pre-trained network')
parser.add_argument('image_file', help = 'image to reconstruct')
parser.add_argument('output_file', help = 'output png file to store')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-n', '--noise_level', type = float, help = 'level of S&P noise', default = 0.0)
parser.add_argument('-i', '--image_size', type = int, help = 'size of the input image', default = 128)
args = parser.parse_args()

# apply seed
if args.seed is not None:
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

# load the model
only_train = True
model = tf.keras.models.load_model(args.network_file, custom_objects={'SaltAndPepper': SaltAndPepper}, compile = False)
for layer in model.layers:
    if hasattr(layer, 'only_train') and getattr(layer, 'only_train') == False:
        only_train = False
        if args.noise_level != getattr(layer, 'ratio'):
            raise Exception("cannot deactivate noise during testing!")


# only need to manually insert noise if keras noise layer is only active during training
if only_train:
    noise_function = lambda x: salt_and_pepper_noise(x, args.noise_level, args.image_size, 1.0)
else:
    noise_function = lambda x: x

# load image
img = cv2.imread(args.image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img / 255
img = cv2.resize(img, (args.image_size, args.image_size))

# collect the reconstruction
images = [noise_function(img)]
model_outputs = model.predict(np.array(images))
results = model_outputs[-1]

# visualize
display_img = results[0]
display_img = display_img.reshape((display_img.shape[0], display_img.shape[1]))
plt.imsave(args.output_file, arr=display_img, cmap="gray")