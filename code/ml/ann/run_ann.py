# -*- coding: utf-8 -*-
"""
Train and evaluate our proposed ANN architecture.

Created on Wed Dec  9 10:53:30 2020

@author: lbechberger
"""

import argparse, pickle, os, fcntl
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
NUM_FOLDS = 5
NUM_CLASSES = 291
TEST_LIMIT = 2000

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

folds = {}

print('loading fold data ...')
for fold in range(NUM_FOLDS):
    
    print(fold)    
    shapes_img = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1))
    shapes_coords = np.zeros((0, space_dim))
    shapes_classes = np.zeros((0, NUM_CLASSES))
    if shapes_data is not None:
        img_list = []
        coords_list = []
        
        for img_path, img_id in shapes_data[str(fold)]:
            
            if args.test and len(img_list) > TEST_LIMIT:
                break
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            img_list.append(img)
            
            coordinates = shapes_targets[img_id]
            coords_list.append(coordinates)
            
        shapes_img = np.array(img_list)
        shapes_img = shapes_img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        shapes_coords = np.array(coords_list)
        shapes_classes = np.zeros((shapes_img.shape[0],NUM_CLASSES))
    
    additional_img = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1))
    additional_coords = np.zeros((0, space_dim))
    additional_classes = np.zeros((0, NUM_CLASSES))
    if additional_data is not None:
        img_list = []

        for img_path, _ in additional_data[str(fold)]:
            
            if args.test and len(img_list) > TEST_LIMIT:
                break

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            img_list.append(img)
        
        additional_img = np.array(img_list)
        additional_img = additional_img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        additional_coords = np.zeros((additional_img.shape[0], space_dim))
        additional_classes = np.zeros((additional_img.shape[0], NUM_CLASSES))
    
    berlin_img = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1))
    berlin_coords = np.zeros((0, space_dim))
    berlin_classes = np.zeros((0, NUM_CLASSES))
    if berlin_data is not None:
        img_list = []
        class_list = []

        for img_path, img_class in berlin_data[str(fold)]:
            
            if args.test and len(img_list) > TEST_LIMIT:
                break

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            img_list.append(img)
            class_list.append(img_class)
        
        berlin_img = np.array(img_list)
        berlin_img = berlin_img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        berlin_classes = class_list
        berlin_coords = np.zeros((berlin_img.shape[0], space_dim))
    
    sketchy_img = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1))
    sketchy_coords = np.zeros((0, space_dim))
    sketchy_classes = np.zeros((0, NUM_CLASSES))
    if sketchy_data is not None:
        img_list = []
        class_list = []

        for img_path, img_class in sketchy_data[str(fold)]:
            
            if args.test and len(img_list) > TEST_LIMIT:
                break

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            img_list.append(img)
            class_list.append(img_class)
        
        sketchy_img = np.array(img_list)
        sketchy_img = sketchy_img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        sketchy_classes = class_list
        sketchy_coords = np.zeros((sketchy_img.shape[0], space_dim))
    
    label_encoder = LabelEncoder()
    all_classes = label_encoder.fit_transform(berlin_classes + sketchy_classes)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(all_classes.reshape(-1, 1))
    berlin_classes = one_hot_encoder.transform(label_encoder.transform(berlin_classes).reshape(-1, 1))
    sketchy_classes = one_hot_encoder.transform(label_encoder.transform(sketchy_classes).reshape(-1, 1))
    
    print('shapes', shapes_img.shape, shapes_coords.shape, shapes_classes.shape)
    print('additional', additional_img.shape, additional_coords.shape, additional_classes.shape)
    print('berlin', berlin_img.shape, berlin_coords.shape, berlin_classes.shape)
    print('sketchy', sketchy_img.shape, sketchy_coords.shape, sketchy_classes.shape)

    if args.reconstruction_weight > 0:
        shapes_proportion = 21
        additional_proportion = 21
        berlin_proportion = 43
        sketchy_proportion = 43
    elif args.mapping_weight > 0:
        shapes_proportion = 26
        additional_proportion = 0
        berlin_proportion = 51
        sketchy_proportion = 51
    else:
        shapes_proportion = 0
        additional_proportion = 0
        berlin_proportion = 64
        sketchy_proportion = 64
    
    fold_img = []
    fold_coords = []
    fold_classes = []

    weights_classification = []
    weights_mapping = []
    weights_reconstruction = []    
    
    counter = 0
    while not ((counter + 1)*shapes_proportion > shapes_coords.shape[0] or (counter + 1)*additional_proportion > additional_coords.shape[0]
            or (counter + 1)*berlin_proportion > berlin_coords.shape[0] or (counter + 1)*sketchy_proportion > sketchy_coords.shape[0]):

        shapes_img_slice = shapes_img[counter * shapes_proportion:(counter + 1) * shapes_proportion]
        shapes_coords_slice = shapes_coords[counter * shapes_proportion:(counter + 1) * shapes_proportion]
        shapes_classes_slice = shapes_classes[counter * shapes_proportion:(counter + 1) * shapes_proportion]
        
        additional_img_slice = additional_img[counter * additional_proportion:(counter + 1) * additional_proportion]
        additional_coords_slice = additional_coords[counter * additional_proportion:(counter + 1) * additional_proportion]
        additional_classes_slice = additional_classes[counter * additional_proportion:(counter + 1) * additional_proportion]
        
        berlin_img_slice = berlin_img[counter * berlin_proportion:(counter + 1) * berlin_proportion]
        berlin_coords_slice = berlin_coords[counter * berlin_proportion:(counter + 1) * berlin_proportion]
        berlin_classes_slice = berlin_classes[counter * berlin_proportion:(counter + 1) * berlin_proportion]
        
        sketchy_img_slice = sketchy_img[counter * sketchy_proportion:(counter + 1) * sketchy_proportion]
        sketchy_coords_slice = sketchy_coords[counter * sketchy_proportion:(counter + 1) * sketchy_proportion]
        sketchy_classes_slice = sketchy_classes[counter * sketchy_proportion:(counter + 1) * sketchy_proportion]
        
        fold_img.append(np.concatenate([shapes_img_slice, additional_img_slice, berlin_img_slice, sketchy_img_slice]))
        fold_coords.append(np.concatenate([shapes_coords_slice, additional_coords_slice, berlin_coords_slice, sketchy_coords_slice]))
        fold_classes.append(np.concatenate([shapes_classes_slice, additional_classes_slice, berlin_classes_slice, sketchy_classes_slice]))
        
        if args.classification_weight > 0:
            weights_classification.append([0]*shapes_coords_slice.shape[0] + [0]*additional_coords_slice.shape[0] + [1]*berlin_coords_slice.shape[0] + [1]*sketchy_coords_slice.shape[0])
        if args.mapping_weight > 0:
            weights_mapping.append([1]*shapes_coords_slice.shape[0] + [0]*additional_coords_slice.shape[0] + [0]*berlin_coords_slice.shape[0] + [0]*sketchy_coords_slice.shape[0])
        if args.reconstruction_weight > 0:
            weights_reconstruction.append([1]*shapes_coords_slice.shape[0] + [1]*additional_coords_slice.shape[0] + [1]*berlin_coords_slice.shape[0] + [1]*sketchy_coords_slice.shape[0])
        
        counter += 1
    
    fold_img = np.concatenate(fold_img)
    fold_coords = np.concatenate(fold_coords)
    fold_classes = np.concatenate(fold_classes)
    weights_classification = np.concatenate(weights_classification)
    weights_mapping = np.concatenate(weights_mapping)
    weights_reconstruction =  np.concatenate(weights_reconstruction)
    
    print(fold_img.shape, fold_coords.shape, fold_classes.shape, weights_classification.shape, weights_mapping.shape, weights_reconstruction.shape)    
    
    folds[fold] = {'img': fold_img, 'mapping': fold_coords, 'classes': fold_classes, 'weights': {'classification': weights_classification, 'mapping': weights_mapping, 'reconstruction': weights_reconstruction}}
    
    
    
# data source provider: load images from respective sources, rescale them to [0,1], iterator returning specified number
# overall batch provider: create data source providers as needed, iterator returns combination of their iterators 

# define network structure

# encoder
enc_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
# TODO: make noise!
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

bottleneck = tf.keras.layers.Concatenate(axis=1, name = 'bottleneck')([enc_mapping, enc_other])

# classifier
class_softmax = tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax', name = 'classification')(bottleneck)

# decoder
dec_fc1 = tf.keras.layers.Dense(512, activation = 'relu')(bottleneck)
dec_d1 = tf.keras.layers.Dropout(0.5)(dec_fc1) if args.decoder_dropout else dec_fc1
dec_fc2 = tf.keras.layers.Dense(4608)(dec_d1)
dec_img = tf.keras.layers.Reshape((3,3,512))(dec_fc2)
dec_uconv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides = 1, activation = 'relu', padding = 'valid')(dec_img)
dec_uconv2 = tf.keras.layers.Conv2DTranspose(256, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv1)
dec_uconv3 = tf.keras.layers.Conv2DTranspose(128, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv2)
dec_uconv4 = tf.keras.layers.Conv2DTranspose(64, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv3)
dec_uconv5 = tf.keras.layers.Conv2DTranspose(32, 5, strides = 2, activation = 'relu', padding = 'same')(dec_uconv4)
dec_output = tf.keras.layers.Conv2DTranspose(1, 5, strides = 2, activation = 'sigmoid', padding = 'same', name = 'reconstruction')(dec_uconv5)

# set up model and loss
model = tf.keras.models.Model(inputs = enc_input, outputs = [class_softmax, enc_mapping, dec_output])
model.compile(optimizer='adam', 
              loss =  {'classification': 'categorical_crossentropy', 'mapping': 'mse', 'reconstruction': 'binary_crossentropy'}, 
              loss_weights = {'classification': args.classification_weight, 'mapping': args.mapping_weight, 'reconstruction': args.reconstruction_weight})
model.summary()


# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
# https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3

#X_train = np.random.uniform(size=(10*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
#y_train = [np.eye(NUM_CLASSES)[np.random.choice(NUM_CLASSES, 10*BATCH_SIZE)], np.random.uniform(size=(10*BATCH_SIZE, space_dim)), 
#           X_train]
#weights_train = {'classification': np.array(([1]*8+[0,0])*BATCH_SIZE), 'mapping': np.array(([0]*8+[1,1])*BATCH_SIZE), 'reconstruction': np.array([1]*10*BATCH_SIZE)}
#
#X_val = np.random.uniform(size=(1*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
#y_val = [np.eye(NUM_CLASSES)[np.random.choice(NUM_CLASSES, 1*BATCH_SIZE)], np.random.uniform(size=(1*BATCH_SIZE, space_dim)), 
#           X_val]
#weights_val = {'classification': np.array([1,0,1,1]*32), 'mapping': np.array(([0,1,0,0])*32), 'reconstruction': np.array([1]*BATCH_SIZE)}
#
#X_test = np.random.uniform(size=(1*BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
#y_test = [np.eye(NUM_CLASSES)[np.random.choice(NUM_CLASSES, 1*BATCH_SIZE)], np.random.uniform(size=(1*BATCH_SIZE, space_dim)), 
#           X_test]

X_train = np.concatenate([folds[0]['img'], folds[1]['img'], folds[2]['img']])
X_val = folds[3]['img']
X_test = folds[4]['img']

y_train = [np.concatenate([folds[0]['classes'], folds[1]['classes'], folds[2]['classes']]),
           np.concatenate([folds[0]['mapping'], folds[1]['mapping'], folds[2]['mapping']]),
           X_train]
y_val = [folds[3]['classes'], folds[3]['mapping'], X_val]
y_test = [folds[4]['classes'], folds[4]['mapping'], X_val]

weights_train = {'classification': np.concatenate([folds[0]['weights']['classification'], folds[1]['weights']['classification'], folds[2]['weights']['classification']]),
                 'mapping': np.concatenate([folds[0]['weights']['mapping'], folds[1]['weights']['mapping'], folds[2]['weights']['mapping']]),
                 'reconstruction': np.concatenate([folds[0]['weights']['reconstruction'], folds[1]['weights']['reconstruction'], folds[2]['weights']['reconstruction']])}
weights_val = folds[3]['weights']
weights_test = folds[4]['weights']
           
early_stopping = tf.keras.callbacks.EarlyStopping()
history = model.fit(X_train, y_train, epochs = 50, batch_size = BATCH_SIZE, 
                    validation_data = (X_val, y_val, weights_val), 
                    callbacks = [early_stopping],
                    sample_weight = weights_train,
                    shuffle = False)

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