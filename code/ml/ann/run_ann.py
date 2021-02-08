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
from sklearn.preprocessing import OneHotEncoder
from code.ml.ann.keras_utils import SaltAndPepper, AutoRestart, EarlyStoppingRestart, IndividualSequence, OverallSequence
from code.util import precompute_distances, compute_correlations, distance_functions

parser = argparse.ArgumentParser(description='Training and evaluating a hybrid ANN')
parser.add_argument('shapes_file', help = 'pickle file containing information about the Shapes data')
parser.add_argument('additional_file', help = 'pickle file containing information about the additional line drawing data')
parser.add_argument('berlin_file', help = 'pickle file containing information about the TU Berlin data')
parser.add_argument('sketchy_file', help = 'pickle file containing information about the Sketchy data')
parser.add_argument('targets_file', help = 'pickle file containing the regression targets')
parser.add_argument('space', help = 'name of the target space to use')
parser.add_argument('image_folder', help = 'folder of original shape images (used for correlation computation)')
parser.add_argument('dissimilarity_file', help = 'pickle file containing the target dissimilarities (used for correlation computation)')
parser.add_argument('output_file', help = 'csv file for outputting the results')
parser.add_argument('-c', '--classification_weight', type = float, help = 'relative weight of classification objective in overall loss function', default = 0.0)
parser.add_argument('-r', '--reconstruction_weight', type = float, help = 'relative weight of reconstruction objective in overall loss function', default = 0.0)
parser.add_argument('-m', '--mapping_weight', type = float, help = 'relative weight of mapping objective in overall loss function', default = 0.0)
parser.add_argument('-b', '--bottleneck_size', type = int, help = 'number of units in the bottleneck layer', default = 512)
parser.add_argument('-w', '--weight_decay_encoder', type = float, help = 'weight decay penalty for encoder', default = 0.0005)
parser.add_argument('-v', '--weight_decay_decoder', type = float, help = 'weight decay penalty for decoder', default = 0.0)
parser.add_argument('-e', '--encoder_dropout', action = 'store_true', help = 'use dropout in encoder')
parser.add_argument('-d', '--decoder_dropout', action = 'store_true', help = 'use dropout in decoder')
parser.add_argument('-n', '--noise_prob', type = float, help = 'probability of salt and pepper noise', default = 0.1)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-t', '--test', action = 'store_true', help = 'make only short test run instead of full training cycle')
parser.add_argument('-f', '--fold', type = int, help = 'fold to use for testing', default = 0)
parser.add_argument('--walltime', type = int, help = 'walltime after which the job will be killed', default = None)
parser.add_argument('--stopped_epoch', type = int, help = 'epoch where the training was interrupted', default = None)
parser.add_argument('--early_stopped', action = 'store_true', help = 'training was stopped with early stopping')
parser.add_argument('--optimizer', help = 'optimizer to use', default = 'adam')
parser.add_argument('--learning_rate', type = float, help = 'learning rate for the optimizer', default = 0.0001)
parser.add_argument('--momentum', type = float, help = 'momentum for the optimizer', default = 0.9)
args = parser.parse_args()

if args.classification_weight + args.reconstruction_weight + args.mapping_weight == 0:
    raise Exception("At least one objective needs to have positive weight!")

start_time = time.time()
    
IMAGE_SIZE = 128
NUM_FOLDS = 5
EPOCHS = 100 if not args.test else 3
initial_epoch = 0 if args.stopped_epoch is None else args.stopped_epoch + 1

# apply seed
if args.seed is not None:
    tf.set_random_seed(args.seed + initial_epoch)
    np.random.seed(args.seed + initial_epoch)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# configuration string
config_name = "c{0}_r{1}_m{2}_b{3}_w{4}_v{5}_e{6}_d{7}_n{8}_{9}".format(
                args.classification_weight, args.reconstruction_weight, args.mapping_weight,
                args.bottleneck_size, args.weight_decay_encoder, args.weight_decay_decoder,
                args.encoder_dropout, args.decoder_dropout, args.noise_prob, args.space)


do_c = args.classification_weight > 0   # should we do a classification?
do_m = args.mapping_weight > 0          # should we do the mapping?
do_r = args.reconstruction_weight > 0   # should we do the reconstruction?

# loads the fold information about a datasubset from a given file
def load_data(file):
    with open(file, 'rb') as f_in:
        data = pickle.load(f_in)
    return data

# loads all data relevant to the Shapes data set
def load_shapes(shapes_file, targets_file):
    shapes_data = load_data(shapes_file)
    with open(targets_file, 'rb') as f_in:
        shapes_targets = pickle.load(f_in)[args.space]['correct']
    space_dim = len(list(shapes_targets.values())[0])
    return shapes_data, shapes_targets, space_dim

# loads data with classification information
def load_classification(file):
    data = load_data(file)
    class_set = set(map(lambda x: x[1], data['0']))
    return data, class_set

# defines a mapping from classes to one-hot-vectors
def get_class_mapping(class_set):
    class_list = list(class_set)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(np.array(class_list).reshape(-1, 1))
    class_map = {}
    for label in class_list:
        one_hot_label = one_hot_encoder.transform(np.array([label]).reshape(-1, 1))
        class_map[label] = one_hot_label.reshape(-1)
    return class_map

# merges two mappings into a single one
def merge_mappings(mappings):
    output_mapping = {}
    
    sizes = []
    for mapping in mappings:
        sizes.append(len(mapping))
    
    for idx, mapping in enumerate(mappings):

        prefix_length = sum([sizes[i] for i in range(idx)])
        prefix = [0]*prefix_length
        suffix_length = sum([sizes[i] for i in range(idx + 1, len(mappings))])
        suffix = [0]*suffix_length
        
        for key, value in mapping.items():
            output_mapping[key] = np.concatenate([prefix, value, suffix])
    
    return output_mapping

# load data as needed
shapes_data = None
additional_data = None
berlin_data = None
sketchy_data = None
shapes_targets = None
space_dim = 0

if do_m or do_r:
    # load Shapes data
    shapes_data, shapes_targets, space_dim = load_shapes(args.shapes_file, args.targets_file)
    
if do_c or do_r:
    # load classification data
    berlin_data, berlin_classes = load_classification(args.berlin_file)
    sketchy_data, sketchy_classes = load_classification(args.sketchy_file)
    
    # bulid the maps for the individual parts
    common_map = get_class_mapping(berlin_classes.intersection(sketchy_classes))
    berlin_only_map = get_class_mapping(berlin_classes.difference(sketchy_classes))
    sketchy_only_map = get_class_mapping(sketchy_classes.difference(berlin_classes))
    
    # merge them appropriately for the respective output layers
    berlin_map = merge_mappings([common_map, berlin_only_map])
    sketchy_map = merge_mappings([common_map, sketchy_only_map])
    overall_map = merge_mappings([common_map, berlin_only_map, sketchy_only_map])

if do_r:
    # load additional data
    additional_data = load_data(args.additional_file)    

with open(args.dissimilarity_file, 'rb') as f_in:
    dissimilarity_data = pickle.load(f_in)

# load original images for evaluation
original_images = []
for item_id in dissimilarity_data['items']:
    for file_name in os.listdir(args.image_folder):
            if os.path.isfile(os.path.join(args.image_folder, file_name)) and item_id in file_name:
                img = cv2.imread(os.path.join(args.image_folder, file_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img / 255
                original_images.append(img)
                break
original_images = np.array(original_images)
original_images = original_images.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
target_dissimilarities = dissimilarity_data['dissimilarities']


# define network structure
def create_model(do_classification, do_mapping, do_reconstruction):
    
    model_outputs = []    
    model_loss = {}
    model_loss_weights = {}
    model_metrics = {}
    
    # encoder
    enc_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    enc_noise = SaltAndPepper(ratio = args.noise_prob)(enc_input)
    enc_conv1 = tf.keras.layers.Conv2D(64, 15, strides = 2, activation = 'relu', padding = 'valid',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_noise)
    enc_mp1 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv1)
    enc_conv2 = tf.keras.layers.Conv2D(128, 5, strides = 1, activation = 'relu', padding = 'valid',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_mp1)
    enc_mp2 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv2)
    enc_conv3 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_mp2)
    enc_conv4 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_conv3)
    enc_conv5 = tf.keras.layers.Conv2D(256, 3, strides = 1, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_conv4)
    enc_mp5 = tf.keras.layers.MaxPool2D(3, 2, padding = 'valid')(enc_conv5)
    enc_flat = tf.keras.layers.Flatten()(enc_mp5)
    enc_fc1 = tf.keras.layers.Dense(512, activation='relu',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_flat)
    enc_d1 = tf.keras.layers.Dropout(0.5)(enc_fc1) if args.encoder_dropout else enc_fc1
    enc_mapping = tf.keras.layers.Dense(space_dim, activation = None, name = 'mapping',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_d1)
    
    if do_mapping:
        model_outputs.append(enc_mapping)

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
    
        def mse(y_true, y_pred):
            return space_dim * tf.keras.losses.mean_squared_error(y_true, y_pred)

        model_loss['mapping'] = mse
        model_loss_weights['mapping'] = args.mapping_weight
        model_metrics['mapping'] = [med, r2]
    
    if do_classification or do_reconstruction:
        enc_other = tf.keras.layers.Dense(args.bottleneck_size - space_dim, activation = None,  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(enc_d1)
        bottleneck = tf.keras.layers.Concatenate(axis = 1, name = 'bottleneck')([enc_mapping, enc_other])
        
        model_outputs.append(bottleneck)
    
    if do_classification:    
        # classifier

        class_dense_berlin = tf.keras.layers.Dense(len(berlin_only_map), activation = None, kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(bottleneck)
        class_dense_sketchy = tf.keras.layers.Dense(len(sketchy_only_map), activation = None, kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(bottleneck)
        class_dense_common = tf.keras.layers.Dense(len(common_map), activation = None, kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_encoder))(bottleneck)
        
        class_berlin = tf.keras.layers.Concatenate(axis = 1)([class_dense_common, class_dense_berlin])
        class_sketchy = tf.keras.layers.Concatenate(axis = 1)([class_dense_common, class_dense_sketchy])
        class_all = tf.keras.layers.Concatenate(axis = 1)([class_dense_common, class_dense_berlin, class_dense_sketchy])
    
        class_berlin_softmax = tf.keras.layers.Activation(activation = 'softmax', name = 'berlin')(class_berlin)
        class_sketchy_softmax = tf.keras.layers.Activation(activation = 'softmax', name = 'sketchy')(class_sketchy)
        class_all_softmax = tf.keras.layers.Activation(activation = 'softmax', name = 'classification')(class_all)
        
        # need to give a loss to 'berlin' and 'sketchy' outputs, otherwise accuracy won't be evaluated
        # add a loss, but fix its weight to zero to make sure it doesn't play a role in practice
        model_outputs.append(class_berlin_softmax)
        model_loss['berlin'] = 'categorical_crossentropy'
        model_loss_weights['berlin'] = 0
        model_metrics['berlin'] = 'accuracy'

        model_outputs.append(class_sketchy_softmax)
        model_loss['sketchy'] = 'categorical_crossentropy'
        model_loss_weights['berlin'] = 0
        model_metrics['sketchy'] = 'accuracy'

        model_outputs.append(class_all_softmax)
        model_loss['classification'] = 'categorical_crossentropy'
        model_loss_weights['classification'] = args.classification_weight
        
    if do_reconstruction:
        # decoder
        dec_fc1 = tf.keras.layers.Dense(512, activation = 'relu',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(bottleneck)
        dec_d1 = tf.keras.layers.Dropout(0.5)(dec_fc1) if args.decoder_dropout else dec_fc1
        dec_fc2 = tf.keras.layers.Dense(2048,  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_d1)
        dec_img = tf.keras.layers.Reshape((4,4,128))(dec_fc2)
        dec_uconv1 = tf.keras.layers.Conv2DTranspose(128, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_img)
        dec_uconv2 = tf.keras.layers.Conv2DTranspose(64, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv1)
        dec_uconv3 = tf.keras.layers.Conv2DTranspose(32, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv2)
        dec_uconv4 = tf.keras.layers.Conv2DTranspose(16, 5, strides = 2, activation = 'relu', padding = 'same',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv3)
        dec_output = tf.keras.layers.Conv2DTranspose(1, 5, strides = 2, activation = 'sigmoid', padding = 'same', name = 'reconstruction',  kernel_regularizer = tf.keras.regularizers.l2(args.weight_decay_decoder))(dec_uconv4)
        
        model_outputs.append(dec_output)   
        model_loss['reconstruction'] = 'binary_crossentropy'
        model_loss_weights['reconstruction'] = args.reconstruction_weight
        
    # set up model, loss, and evaluation metrics
    model = tf.keras.models.Model(inputs = enc_input, outputs = model_outputs)
    
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr = args.learning_rate, momentum = args.momentum)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr = args.learning_rate, beta_1 = args.momentum)
        
    model.compile(optimizer = optimizer, loss = model_loss, loss_weights = model_loss_weights, weighted_metrics = model_metrics)
    #model.summary()
    
    return model


# define data iterators
def get_data_sequence(list_of_folds, do_classification, do_mapping, do_reconstruction, shuffle = True, truncate = True):

    if do_reconstruction:
        shapes_proportion = 21
        additional_proportion = 24
        berlin_proportion = 41
        sketchy_proportion = 42
    elif do_mapping and do_classification:
        shapes_proportion = 25
        additional_proportion = 0
        berlin_proportion = 51
        sketchy_proportion = 52
    elif do_mapping:
        shapes_proportion = 128
        additional_proportion = 0
        berlin_proportion = 0
        sketchy_proportion = 0
    else: # only classification
        shapes_proportion = 0
        additional_proportion = 0
        berlin_proportion = 63
        sketchy_proportion = 65

    seqs = []
    weights = []
    all_classes = 0
    berlin_classes = 0
    sketchy_classes = 0
    
    if shapes_proportion > 0:
        shapes_sequence = IndividualSequence(np.concatenate([shapes_data[str(i)] for i in list_of_folds]), 
                                             [shapes_targets], shapes_proportion, IMAGE_SIZE, shuffle = shuffle, truncate = truncate)
        shapes_weights = {'classification': 0.0, 'mapping': 1.0, 'reconstruction': 1.0, 'berlin': 0.0, 'sketchy': 0.0}
        
        seqs.append(shapes_sequence)
        weights.append(shapes_weights)
    
    if additional_proportion > 0:
        additional_sequence = IndividualSequence(np.concatenate([additional_data[str(i)] for i in list_of_folds]), 
                                                 [{None: 0}], additional_proportion, IMAGE_SIZE, shuffle = shuffle, truncate = truncate)
        additional_weights = {'classification': 0.0, 'mapping': 0.0, 'reconstruction': 1.0, 'berlin': 0.0, 'sketchy': 0.0}
        
        seqs.append(additional_sequence)
        weights.append(additional_weights)
    
    if berlin_proportion > 0:
        berlin_sequence = IndividualSequence(np.concatenate([berlin_data[str(i)] for i in list_of_folds]), 
                                             [overall_map, berlin_map], berlin_proportion, IMAGE_SIZE, shuffle = shuffle, truncate = truncate)
        berlin_weights = {'classification': 1.0, 'mapping': 0.0, 'reconstruction': 1.0, 'berlin': 1.0, 'sketchy': 0.0}

        seqs.append(berlin_sequence)
        weights.append(berlin_weights)
        
        all_classes = len(overall_map)
        berlin_classes = len(berlin_map)
    
    if sketchy_proportion > 0:
        sketchy_sequence = IndividualSequence(np.concatenate([sketchy_data[str(i)] for i in list_of_folds]), 
                                              [overall_map, sketchy_map], sketchy_proportion, IMAGE_SIZE, shuffle = shuffle, truncate = truncate)
        sketchy_weights = {'classification': 1.0, 'mapping': 0.0, 'reconstruction': 1.0, 'berlin': 0.0, 'sketchy': 1.0}
        
        seqs.append(sketchy_sequence)
        weights.append(sketchy_weights)
        
        sketchy_classes = len(sketchy_map)

    data_sequence = OverallSequence(seqs, weights, space_dim, all_classes, berlin_classes, sketchy_classes,
                                    do_classification, do_mapping, do_reconstruction, truncate = truncate)
    return data_sequence


test_fold = args.fold
val_fold = (test_fold - 1) % NUM_FOLDS
train_folds = [i for i in range(NUM_FOLDS) if i != test_fold and i != val_fold]

train_seq = get_data_sequence(train_folds, do_c, do_m, do_r, shuffle = True, truncate = True)
train_steps = len(train_seq) if not args.test else 1
val_seq = get_data_sequence([val_fold], do_c, do_m, do_r, shuffle = False, truncate = False)
val_steps = len(val_seq) if not args.test else 1
test_seq = get_data_sequence([test_fold], do_c, do_m, do_r, shuffle = False, truncate = False)
test_steps = len(test_seq) if not args.test else 1


# set up the callbacks
log_path = os.path.join(os.path.split(args.output_file)[0], 'logs', '{0}_f{1}_log.csv'.format(config_name, args.fold))
storage_path = 'data/Shapes/ml/snapshots/{0}_f{1}_ep'.format(config_name, args.fold)

callbacks = []
csv_logger = tf.keras.callbacks.CSVLogger(log_path, append = True)
callbacks.append(csv_logger)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(storage_path + '{epoch}.hdf5', save_weights_only = True)
callbacks.append(model_checkpoint)

monitor = 'val_mapping_loss' if do_m else 'val_loss'
early_stopping = EarlyStoppingRestart(logpath = log_path, initial_epoch = initial_epoch,
                                      modelpath = storage_path, verbose = 1,
                                      monitor = monitor, patience = 10)
callbacks.append(early_stopping)
if args.walltime is not None:
    auto_restart = AutoRestart(filepath=storage_path, start_time=start_time, verbose = 1, walltime=args.walltime)
    callbacks.append(auto_restart)

# set up the model
model = create_model(do_c, do_m, do_r)
if args.stopped_epoch is not None:
    # later run: load weights from file
    model.load_weights(storage_path + str(args.stopped_epoch) + '.hdf5')
    model._make_train_function()
    with open(storage_path + str(args.stopped_epoch) + '.hdf5.opt', 'rb') as f_in:
        opt_state = pickle.load(f_in)
    model.optimizer.set_weights(opt_state)
    
if not args.early_stopped:
    # train if not already killed by early stopping
    history = model.fit_generator(generator = train_seq, steps_per_epoch = train_steps, epochs = EPOCHS, 
                                  validation_data = val_seq, validation_steps = val_steps,
                                  callbacks = callbacks, shuffle = True, initial_epoch = initial_epoch)

    # interrupted by early stopping or wall time or ran out of epochs --> restart
    from subprocess import call
    recall_list = ['qsub', 'code/ml/ann/run_ann.sge',
                       args.shapes_file, args.additional_file, args.berlin_file, args.sketchy_file,
                       args.targets_file, args.space, args.image_folder, args.dissimilarity_file, args.output_file]
    recall_list += ['-c', str(args.classification_weight), '-r', str(args.reconstruction_weight), '-m', str(args.mapping_weight)]
    recall_list += ['-b', str(args.bottleneck_size), '-w', str(args.weight_decay_encoder), '-v', str(args.weight_decay_decoder)]
    if args.encoder_dropout:
        recall_list += ['-e']
    if args.decoder_dropout:
        recall_list += ['-d']
    recall_list += ['-n', str(args.noise_prob), '-s', str(args.seed)]
    recall_list += ['-f', str(args.fold)]

    recall_list += ['--optimizer', args.optimizer]
    recall_list += ['--learning_rate', str(args.learning_rate)]
    recall_list += ['--momentum', str(args.momentum)]

    if (args.walltime is not None and auto_restart.reached_walltime == 1):
        # stopped by walltime:
        # use updated weights for continued training in next round
        recall_list += ['--walltime', str(args.walltime), '--stopped_epoch', str(auto_restart.stopped_epoch)]
    else:
        # stopped by early stopping or by end of epochs:
        # use best weights for evaluation in next round
        recall_list += ['--early_stopped', '--stopped_epoch', str(early_stopping.best_epoch)]
    
    recall_string = ' '.join(recall_list)
    print(recall_string)
    call(recall_string, shell = True)
else:
    
    # do the evaluation
    evaluation_metrics = []
    evaluation_results = []

    if do_c or do_r:
        # compute overall kendall correlation of bottleneck activation to dissimilarity ratings
        model_outputs = model.predict(original_images)
        bottleneck_activation = model_outputs[1] if do_m else model_outputs[0]
        
        for distance_function in sorted(distance_functions.keys()):
            precomputed_distances = precompute_distances(bottleneck_activation, distance_function)
            kendall_fixed  = compute_correlations(precomputed_distances, target_dissimilarities, distance_function)['kendall']
            kendall_optimized = compute_correlations(precomputed_distances, target_dissimilarities, distance_function, 5, args.seed)['kendall']
            evaluation_metrics += ['kendall_{0}_fixed'.format(distance_function), 'kendall_{0}_optimized'.format(distance_function)]
            evaluation_results += [kendall_fixed, kendall_optimized]

    # compute standard evaluation metrics on the test set
    eval_test = model.evaluate_generator(test_seq, steps = test_steps)
    
    for metric_value, metric_name in zip(eval_test, model.metrics_names):
        evaluation_metrics.append(metric_name)
        evaluation_results.append(metric_value)
        print(metric_name, metric_value)
    
    # prepare output file if necessary
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f_out:
            fcntl.flock(f_out, fcntl.LOCK_EX)
            f_out.write("configuration,fold,{0}\n".format(','.join(evaluation_metrics)))
            fcntl.flock(f_out, fcntl.LOCK_UN)
    
    with open(args.output_file, 'a') as f_out:
        fcntl.flock(f_out, fcntl.LOCK_EX)
        f_out.write("{0},{1},{2}\n".format(config_name, args.fold, ','.join(map(lambda x: str(x), evaluation_results))))
        fcntl.flock(f_out, fcntl.LOCK_UN)

    # store final model: structure + weights (different extension so we don't delete it by accident)
    model.save(storage_path + str(initial_epoch - 1) + '_FINAL.h5', include_optimizer = False)
        
    # remove the old snapshots to free some disk space
    from subprocess import call
    call('rm {0}*.hdf5*'.format(storage_path), shell = True)