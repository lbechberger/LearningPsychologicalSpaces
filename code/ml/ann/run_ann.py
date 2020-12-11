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
        shapes_targets = pickle.load(f_in)[args.space]['correct']
    space_dim = len(list(shapes_targets.values())[0])
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
            shapes_targets = pickle.load(f_in)[args.space]['correct']
        space_dim = len(list(shapes_targets.values())[0])

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
enc_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
enc_conv1 = tf.keras.layers.Conv2D(64, 15, strides = 3, activation = 'relu', padding = 'valid')(enc_input)
enc_mp1 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv1)
enc_conv2 = tf.keras.layers.Conv2D(128, 5, strides = 1, activation = 'relu', padding = 'valid')(enc_mp1)
enc_mp2 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv2)
enc_conv3 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same')(enc_mp2)
enc_conv4 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same')(enc_conv3)
enc_conv5 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same')(enc_conv4)
enc_mp5 = tf.keras.layers.MaxPool2D(3, 2, padding = 'same')(enc_conv5)
enc_flat = tf.keras.layers.Flatten()(enc_mp5)
enc_fc1 = tf.keras.layers.Dense(512, activation='relu')(enc_flat)
enc_d1 = tf.keras.layers.Dropout(0.5)(enc_fc1) if args.encoder_dropout else enc_fc1
enc_mapping = tf.keras.layers.Dense(space_dim, activation=None, name = 'mapping')(enc_d1)
enc_other = tf.keras.layers.Dense(args.bottleneck_size - space_dim, activation=None)(enc_d1)

bottleneck = tf.keras.layers.Concatenate(axis=1)([enc_mapping, enc_other], name = 'bottleneck')

# classifier
class_softmax = tf.keras.layers.Dense(275, activation = 'softmax', name = 'classification')(bottleneck)

# decoder
dec_fc1 = tf.keras.layers.Dense(512, activation = 'relu')(bottleneck)
dec_d1 = tf.keras.layers.Dropout(0.5)(dec_fc1) if args.decoder_dropout else dec_fc1
dec_fc2 = tf.keras.layers.Dense(4096)(dec_d1)
dec_d2 = tf.keras.layers.Dropout(0.5)(dec_fc2) if args.decoder_dropout else dec_fc2
dec_fc3 = tf.keras.layers.Dense(12544)(dec_d2)
dec_img = tf.keras.layers.Reshape((7,7,256))(dec_fc3)
dec_uconv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides = 2, activation = 'relu', padding = 'same')(dec_img)
dec_uconv2 = tf.keras.layers.Conv2DTranspose(128, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv1)
dec_uconv3 = tf.keras.layers.Conv2DTranspose(64, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv2)
dec_uconv4 = tf.keras.layers.Conv2DTranspose(32, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv3)
dec_output = tf.keras.layers.Conv2DTranspose(1, 5, strides = 2, activation = 'sigmoid', padding = 'same', name = 'reconstruction')(dec_uconv4)

# set up model and loss
model = tf.keras.models.Model(inputs = enc_input, outputs = [class_softmax, enc_mapping, dec_output])
model.compile(optimizer='adam', 
              loss =  {'classification': 'categorical_crossentropy', 'mapping': 'mse', 'reconstruction': 'binary_crossentropy'}, 
              loss_weights = {'classification': args.classification_weight, 'mapping': args.mapping_weight, 'reconstruction': args.reconstruction_weight},
              metrics={'classification': 'accuracy'})
model.summary()


# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
# https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3

X_train = np.random.uniform(size=(10*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
y_train = [np.eye(275)[np.random.choice(275, 10*BATCH_SIZE)], np.random.uniform(size=(10*BATCH_SIZE, space_dim)), 
           X_train]
weights_train = {'classification': np.array(([1]*8+[0,0])*BATCH_SIZE), 'mapping': np.array(([0]*8+[1,1])*BATCH_SIZE), 'reconstruction': np.array([1]*10*BATCH_SIZE)}

X_val = np.random.uniform(size=(1*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
y_val = [np.eye(275)[np.random.choice(275, 1*BATCH_SIZE)], np.random.uniform(size=(1*BATCH_SIZE, space_dim)), 
           X_val]
weights_val = {'classification': np.array([1,0,1,1]*32), 'mapping': np.array(([0,1,0,0])*32), 'reconstruction': np.array([1]*BATCH_SIZE)}

X_test = np.random.uniform(size=(1*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
y_test = [np.eye(275)[np.random.choice(275, 1*BATCH_SIZE)], np.random.uniform(size=(1*BATCH_SIZE, space_dim)), 
           X_test]
           
early_stopping = tf.keras.callbacks.EarlyStopping()
history = model.fit(X_train, y_train, epochs = 50, batch_size = BATCH_SIZE, 
                    validation_data = (X_val, y_val, weights_val), 
                    callbacks = [early_stopping],
                    sample_weight = weights_train)

predictions = model.predict(X_test, batch_size = BATCH_SIZE)


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