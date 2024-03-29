# -*- coding: utf-8 -*-
"""
Exports the bottleneck activations of the given pre-trained ANN for the given images
and stores them for later use with a linear regression.

Created on Thu Jan 21 08:04:12 2021

@author: lbechberger
"""

import argparse, pickle
import tensorflow as tf
import numpy as np
from code.ml.ann.keras_utils import IndividualSequence, SaltAndPepper
from code.util import salt_and_pepper_noise

parser = argparse.ArgumentParser(description='Extracting bottleneck activations')
parser.add_argument('shapes_file', help = 'pickle file containing information about the Shapes data')
parser.add_argument('network_file', help = 'hdf5 file containing the pre-trained network')
parser.add_argument('output_file', help = 'pickle file for outputting the results')
parser.add_argument('-m', '--mapping_used', action = 'store_true', help = 'has the network been trained with the mapping objective?')
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-n', '--noise_level', type = float, help = 'level of S&P noise', default = 0.0)
parser.add_argument('-i', '--image_size', type = int, help = 'size of the input image', default = 128)
args = parser.parse_args()

NUM_FOLDS = 5
IMAGE_SIZE = args.image_size
BATCH_SIZE = 128

# apply seed
if args.seed is not None:
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)


# load list of augmented shape images
with open(args.shapes_file, 'rb') as f_in:
    shapes_data = pickle.load(f_in)

# load the model
only_train = True
model = tf.keras.models.load_model(args.network_file, custom_objects={'SaltAndPepper': SaltAndPepper}, compile = False)
for layer in model.layers:
    if hasattr(layer, 'only_train') and getattr(layer, 'only_train') == False:
        only_train = False
        if args.noise_level != getattr(layer, 'ratio'):
            raise Exception("cannot deactivate noise during testing!")

# restructure the data
all_folds = np.concatenate([shapes_data[str(i)] for i in range(NUM_FOLDS)])
data_by_label = {}
for path, label in all_folds:
    if label not in data_by_label:
        data_by_label[label] = []
    data_by_label[label].append((path, 0))

list_of_labels = sorted(data_by_label.keys())


# only need to manually insert noise if keras noise layer is only active during training
if only_train:
    noise_function = lambda x: salt_and_pepper_noise(x, args.noise_level, args.image_size, 1.0)
else:
    noise_function = lambda x: x

# collect the activations
result = {}
for label in list_of_labels:
    data = data_by_label[label]
    print(label, len(data))
    data_seq = IndividualSequence(np.array(data), [{'0': 0}], BATCH_SIZE, IMAGE_SIZE, shuffle = False, truncate = False, mapping_function = noise_function)
    model_outputs = model.predict_generator(data_seq, steps = len(data_seq))
    bottleneck_activation = model_outputs[1] if args.mapping_used else model_outputs[0]
    if len(bottleneck_activation) != len(data):
        bottleneck_activation = np.concatenate(bottleneck_activation)
    result[label] = bottleneck_activation.tolist()

# store them
pickle.dump(result, open(args.output_file, 'wb'))
