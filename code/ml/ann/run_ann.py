# -*- coding: utf-8 -*-
"""
Train and evaluate our proposed ANN architecture.

Created on Wed Dec  9 10:53:30 2020

@author: lbechberger
"""

import argparse, pickle, os, fcntl
import tensorflow as tf
import numpy as np

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
parser.add_argument('-t', '--test', action = 'store_true', help = 'make only short test run instead of full training cycle')
args = parser.parse_args()

if args.classification_weight + args.reconstruction_weight + args.mapping_weight != 1:
    raise Exception("Relative weights of objectives need to sum to one!")

IMAGE_SIZE = 224
BATCH_SIZE = 128

# apply seed
if args.seed is not None:
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

# load data as needed
shapes_data = None
additional_data = None
berlin_data = None
sketchy_data = None
shapes_targets = None

if args.reconstruction_weight > 0:
    # load all
    with open(args.shapes_file, 'rb') as f_in:
        shapes_data = pickle.load(f_in)
    with open(args.targets_file, 'rb') as f_in:
        shapes_targets = pickle.load(f_in)[args.space]
    with open(args.additional_file, 'rb') as f_in:
        additional_data = pickle.load(f_in)
    with open(args.berlin_file, 'rb') as f_in:
        berlin_data = pickle.load(f_in)
    with open(args.sketchy_file, 'rb') as f_in:
        sketchy_data = pickle.load(f_in)
else:
    if args.classification_weight > 0:
        # load berlin and sketchy
        with open(args.berlin_file, 'rb') as f_in:
            berlin_data = pickle.load(f_in)
        with open(args.sketchy_file, 'rb') as f_in:
            sketchy_data = pickle.load(f_in)

    if args.mapping_weight > 0:
        # load shapes
        with open(args.shapes_file, 'rb') as f_in:
            shapes_data = pickle.load(f_in)
        with open(args.targets_file, 'rb') as f_in:
            shapes_targets = pickle.load(f_in)[args.space]

# evaluation metrics to compute and record
evaluation_metrics = []

def add_eval_metric(metric_name):
    for suffix in ['_train', '_val', '_test']:
        evaluation_metrics.append(metric_name + suffix)

add_eval_metric('kendall')
if args.reconstruction_weight > 0:
    add_eval_metric('reconstruction')
if args.classification_weight > 0:
    add_eval_metric('acc_Berlin')
    add_eval_metric('acc_Sketchy')
if args.mapping_weight > 0:
    add_eval_metric('mse')
    add_eval_metric('med')
    add_eval_metric('r2')

# prepare output file if necessary
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("configuration,{0}\n".format(','.join(evaluation_metrics)))
        fcntl.flock(f, fcntl.LOCK_UN)


# data source provider: load images from respective sources, rescale them to [0,1], iterator returning specified number
# overall batch provider: create data source providers as needed, iterator returns combination of their iterators 

# define network structure

# encoder
with tf.contrib.framework.arg_scope([tf.layers.dense, tf.layers.conv2d], activation = tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(args.weight_decay_encoder)):
    enc_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
    enc_noisy = None
    enc_conv1 = tf.layers.conv2d(enc_input, 64, 15, strides = 3, padding = 'valid')
    enc_mp1 = tf.layers.max_pooling2d(enc_conv1, 3, 2, padding = 'valid')
    enc_conv2 = tf.layers.conv2d(enc_mp1, 128, 5, strides = 1, padding = 'valid')
    enc_mp2 = tf.layers.max_pooling2d(enc_conv2, 3, 2, padding = 'valid')
    enc_conv3 = tf.layers.conv2d(enc_mp2, 256, 3, strides = 1, padding = 'same')
    enc_conv4 = tf.layers.conv2d(enc_conv3, 256, 3, strides = 1, padding = 'same')
    enc_conv5 = tf.layers.conv2d(enc_conv4, 256, 3, strides = 1, padding = 'same')
    enc_mp5 = tf.layers.max_pooling2d(enc_conv5, 3, 2, padding = 'valid')
    enc_flat = tf.reshape(enc_mp5, [-1, 12544])
    enc_fc1 = tf.layers.dense(enc_flat, 512)
    enc_d1 = tf.layers.dropout(enc_fc1, rate = 0.5) if args.encoder_dropout else enc_fc1
    bottleneck = tf.layers.dense(enc_d1, args.bottleneck_size, activation = None)

# classifier
class_softmax = tf.layers.dense(bottleneck, 275, activation = tf.nn.relu)

# decoder
with tf.contrib.framework.arg_scope([tf.layers.dense, tf.layers.conv2d_transpose], activation = tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(args.weight_decay_decoder)):
    dec_fc1 = tf.layers.dense(bottleneck, 512)
    dec_d1 = tf.layers.dropout(dec_fc1, rate = 0.5) if args.decoder_dropout else dec_fc1
    dec_fc2 = tf.layers.dense(dec_d1, 4096)
    dec_d2 = tf.layers.dropout(dec_fc2, rate = 0.5) if args.decoder_dropout else dec_fc2
    dec_fc3 = tf.layers.dense(dec_d1, 12544)
    dec_img = tf.reshape(dec_fc3, [-1, 7, 7, 256])
    dec_uconv1 = tf.layers.conv2d_transpose(dec_img, 256, 5, strides = 2)
    dec_uconv2 = tf.layers.conv2d_transpose(dec_uconv1, 128, 5, strides = 2)
    dec_uconv3 = tf.layers.conv2d_transpose(dec_uconv2, 64, 5, strides = 2)
    dec_uconv4 = tf.layers.conv2d_transpose(dec_uconv3, 32, 5, strides = 2)
    dec_output = tf.layers.conv2d_transpose(dec_uconv4, 1, 5, strides = 2, activation = tf.nn.sigmoid)

# define loss
classification_loss = tf.losses.softmax_cross_entropy(onehot_labels, class_softmax)
reconstruction_loss = tf.losses.sigmoid_cross_entropy(enc_input, dec_output)
mapping_loss = tf.losses.mean_squared_error(target_coordinates, bottleneck[:,:4])

overall_loss = (args.classification_weight * classification_loss 
                + args.reconstruction_weight * reconstruction_loss
                + args.mapping_weight * mapping_loss)

train_step = tf.train.AdamOptimizer().minimize(overall_loss)

# https://www.oreilly.com/content/building-deep-learning-neural-networks-using-tensorflow-layers/

# cross-validation loop

# for test_fold in range(5):
#   valid_fold = (test_fold - 1) % 5
#   train_folds = all others

#   create a new batch provider for each data subset

#   training loop:
#   with tf.Session() as sess:
#       initialize all variables
#       train with Adam (early stopping: check validation set performance every epoch)
#       when done: evaluate on train, valid, test; store results

# aggregate results across folds, output them