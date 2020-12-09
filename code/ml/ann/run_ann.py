# -*- coding: utf-8 -*-
"""
Train and evaluate our proposed ANN architecture.

Created on Wed Dec  9 10:53:30 2020

@author: lbechberger
"""

import argparse, pickle
import tensorflow as tf

parser = argparse.ArgumentParser(description='Training and evaluating a hybrid ANN')
parser.add_argument('shapes_file', help = 'pickle file containing information about the Shapes data')
parser.add_argument('additional_file', help = 'pickle file containing information about the additional line drawing data')
parser.add_argument('berlin_file', help = 'pickle file containing information about the TU Berlin data')
parser.add_argument('sketchy_file', help = 'pickle file containing information about the Sketchy data')
parser.add_argument('targets_file', help = 'pickle file containing the regression targets')
parser.add_argument('space', help = 'name of the target space to use')
parser.add_argument('output_file', help = 'csv file for outputting the results')
parser.add_argument('-c', '--classification_weight', type = float, help = 'relative weight of classification objective in overall loss function', default = 0)
parser.add_argument('-r', '--reconstruction_weight', type = float, help = 'relative weight of reconstruction objective in overall loss function', default = 0)
parser.add_argument('-m', '--mapping_weight', type = float, help = 'relative weight of mapping objective in overall loss function', default = 0)
parser.add_argument('-b', '--bottleneck_size', type = int, help = 'number of units in the bottleneck layer', default = 512)
parser.add_argument('-w', '--weight_decay_encoder', type = float, help = 'weight decay penalty for encoder', default = 0.0005)
parser.add_argument('-v', '--weight_decay_decoder', type = float, help = 'weight decay penalty for decoder', default = 0)
parser.add_argument('-e', '--encoder_dropout', action = 'store_true', help = 'use dropout in encoder')
parser.add_argument('-d', '--decoder_dropout', action = 'store_true', help = 'use dropout in decoder')
parser.add_argument('-n', '--noise_prob', type = float, help = 'probability of salt and pepper noise', default = 0.1)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
args = parser.parse_args()

if args.classification_weight + args.reconstruction_weight + args.mapping_weight != 1:
    raise Exception("Relative weights of objectives need to sum to one!")

print(args)