# -*- coding: utf-8 -*-
"""
Train and evaluate our proposed ANN architecture.

Created on Wed Dec  9 10:53:30 2020

@author: lbechberger
"""

import argparse, pickle, os, fcntl, time
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from code.ml.ann.keras_utils import SaltAndPepper, AutoRestart

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
parser.add_argument('-f', '--fold', type = int, help = 'fold to use for testing', default = 0)
parser.add_argument('--walltime', type = int, help = 'walltime after which the job will be killed', default = None)
parser.add_argument('--stopped_epoch', type = int, help = 'epoch where the training was interrupted', default = 0)
args = parser.parse_args()

if args.classification_weight + args.reconstruction_weight + args.mapping_weight != 1:
    raise Exception("Relative weights of objectives need to sum to one!")

start_time = time.time()
    
IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_FOLDS = 5
NUM_CLASSES = 291
TEST_LIMIT = 2000
EPOCHS = 100 if not args.test else 1

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
evaluation_metrics = ['kendall']
results = {}

def add_eval_metric(metric_name):
    for suffix in ['_train', '_val', '_test']:
        evaluation_metrics.append(metric_name + suffix)
        results[metric_name + suffix] = 0

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
        f.write("configuration,fold,{0}\n".format(','.join(evaluation_metrics)))
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
            
            if args.test and len(img_list) >= TEST_LIMIT:
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
            
            if args.test and len(img_list) >= TEST_LIMIT:
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
            
            if args.test and len(img_list) >= TEST_LIMIT:
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
            
            if args.test and len(img_list) >= TEST_LIMIT:
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
def create_model():
    # encoder
    enc_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    enc_noise = SaltAndPepper(ratio = args.noise_prob)(enc_input)
    enc_conv1 = tf.keras.layers.Conv2D(64, 15, strides = 3, activation = 'relu', padding = 'valid',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_noise)
    enc_mp1 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv1)
    enc_conv2 = tf.keras.layers.Conv2D(128, 5, strides = 1, activation = 'relu', padding = 'valid',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_mp1)
    enc_mp2 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv2)
    enc_conv3 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_mp2)
    enc_conv4 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_conv3)
    enc_conv5 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_conv4)
    enc_mp5 = tf.keras.layers.MaxPool2D(3, 2, padding = 'same')(enc_conv5)
    enc_flat = tf.keras.layers.Flatten()(enc_mp5)
    enc_fc1 = tf.keras.layers.Dense(512, activation='relu',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_flat)
    enc_d1 = tf.keras.layers.Dropout(0.5)(enc_fc1) if args.encoder_dropout else enc_fc1
    enc_mapping = tf.keras.layers.Dense(space_dim, activation = None, name = 'mapping',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_d1)
    enc_other = tf.keras.layers.Dense(args.bottleneck_size - space_dim, activation = None,  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_d1)
    
    bottleneck = tf.keras.layers.Concatenate(axis=1, name = 'bottleneck')([enc_mapping, enc_other])
    
    # classifier
    class_softmax = tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax', name = 'classification',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(bottleneck)
    
    # decoder
    dec_fc1 = tf.keras.layers.Dense(512, activation = 'relu',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(bottleneck)
    dec_d1 = tf.keras.layers.Dropout(0.5)(dec_fc1) if args.decoder_dropout else dec_fc1
    dec_fc2 = tf.keras.layers.Dense(4608,  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_d1)
    dec_img = tf.keras.layers.Reshape((3,3,512))(dec_fc2)
    dec_uconv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides = 1, activation = 'relu', padding = 'valid',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_img)
    dec_uconv2 = tf.keras.layers.Conv2DTranspose(256, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv1)
    dec_uconv3 = tf.keras.layers.Conv2DTranspose(128, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv2)
    dec_uconv4 = tf.keras.layers.Conv2DTranspose(64, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv3)
    dec_uconv5 = tf.keras.layers.Conv2DTranspose(32, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv4)
    dec_output = tf.keras.layers.Conv2DTranspose(1, 5, strides = 2, activation = 'sigmoid', padding = 'same', name = 'reconstruction',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv5)
    
    # set up model, loss, and evaluation metrics
    model = tf.keras.models.Model(inputs = enc_input, outputs = [class_softmax, enc_mapping, dec_output])

    def r2(y_true, y_pred):
        residual = tf.reduce_sum(tf.square(y_true - y_pred), axis = 1)
        total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis = 0)), axis = 1)
        result = (1 - residual/(total + tf.keras.backend.epsilon()))
        return result
    
    def med(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        sum_squared_diff = tf.reduce_sum(squared_diff, axis = 1)
        eucl_dist = tf.sqrt(sum_squared_diff)
        return eucl_dist
    
    model.compile(optimizer='adam', 
                  loss =  {'classification': 'categorical_crossentropy', 'mapping': 'mse', 'reconstruction': 'binary_crossentropy'}, 
                  loss_weights = {'classification': args.classification_weight, 'mapping': args.mapping_weight, 'reconstruction': args.reconstruction_weight},
                  weighted_metrics = {'classification': 'accuracy', 'mapping': [med,r2]})
    #model.summary()
    
    return model


# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
# https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3



test_fold = args.fold
val_fold = (test_fold - 1) % NUM_FOLDS
train_folds = [i for i in range(NUM_FOLDS) if i != test_fold and i != val_fold]

# prepare data
X_train = np.concatenate([folds[i]['img'] for i in train_folds])
X_val = folds[val_fold]['img']
X_test = folds[test_fold]['img']

y_train = [np.concatenate([folds[i]['classes'] for i in train_folds]),
           np.concatenate([folds[i]['mapping'] for i in train_folds]),
           X_train]
y_val = [folds[val_fold]['classes'], folds[val_fold]['mapping'], X_val]
y_test = [folds[test_fold]['classes'], folds[test_fold]['mapping'], X_val]

weights_train = {'classification': np.concatenate([folds[i]['weights']['classification'] for i in train_folds]),
                 'mapping': np.concatenate([folds[i]['weights']['mapping'] for i in train_folds]),
                 'reconstruction': np.concatenate([folds[i]['weights']['reconstruction'] for i in train_folds])}
weights_val = folds[val_fold]['weights']
weights_test = folds[test_fold]['weights']

# set up the model    
model = create_model()
callbacks = [tf.keras.callbacks.EarlyStopping()]
if args.walltime is not None:
    auto_restart = AutoRestart(filepath='walltime_epoch', start_time=start_time, verbose = 0, walltime=args.walltime)
    callbacks.append(auto_restart)

if args.stopped_epoch > 0:
    model.load_weights('walltime_epoch' + str(args.stopped_epoch) + '.hdf5')

# train it
history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                    validation_data = (X_val, y_val, weights_val), 
                    callbacks = callbacks,
                    sample_weight = weights_train,
                    shuffle = False)

if args.walltime is not None and auto_restart.reached_walltime == 1:
    # interrupted by wall time --> restart
    from subprocess import call
    recall_list = ['qsub', 'run_ann.sge', 
                       args.shapes_file, args.additional_file, args.berlin_file, args.sketchy_file,
                       args.targets_file, args.space, args.output_file]
    recall_list += ['-c', str(args.classification_weight), '-r', str(args.reconstruction_weight), '-m', str(args.mapping_weight)]
    recall_list += ['-b', str(args.bottleneck_size), '-w', str(args.weight_decay_encoder), '-v', str(args.weight_decay_decoder)]
    if args.encoder_dropout:
        recall_list += ['-e']
    if args.decoder_dropout:
        recall_list += ['-d']
    recall_list += ['-n', str(args.noise_prob), '-s', str(args.seed)]
    recall_list += ['--walltime', str(args.walltime), '--stopped_epoch', str(auto_restart.stopped_epoch)]
    
    recall_string = ' '.join(recall_list)    
    call(recall_string, shell = True)
else:
    # do the evaluation
    eval_train = model.evaluate(X_train, y_train, sample_weight = weights_train, batch_size = BATCH_SIZE)
    eval_val = model.evaluate(X_val, y_val, sample_weight = weights_val, batch_size = BATCH_SIZE)
    eval_test = model.evaluate(X_test, y_test, sample_weight = weights_test, batch_size = BATCH_SIZE)
    
    # TODO: overall correlation to dissimilarity ratings    
    
    for output, label in zip(eval_test, model.metrics_names):
        print(label, output)
    
    # TODO: output results to csv
