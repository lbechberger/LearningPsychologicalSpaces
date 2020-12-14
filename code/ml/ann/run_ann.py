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
from code.ml.ann.keras_utils import SaltAndPepper, AutoRestart, IndividualSequence, OverallSequence
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

# configuration string
config_name = "c{0}_r{1}_m{2}_b{3}_w{4}_v{5}_e{6}_d{7}_n{8}_{9}".format(
                args.classification_weight, args.reconstruction_weight, args.mapping_weight,
                args.bottleneck_size, args.weight_decay_encoder, args.weight_decay_decoder,
                args.encoder_dropout, args.decoder_dropout, args.noise_prob, args.space)

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
    model = tf.keras.models.Model(inputs = enc_input, outputs = [class_softmax, enc_mapping, dec_output, bottleneck])

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


# define data iterators
def get_data_sequence(list_of_folds):

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

    # defining a mapping from classes to one-hot-vectors
    class_map = {}
    berlin_classes = set(map(lambda x: x[1], berlin_data['0']))
    sketchy_classes = set(map(lambda x: x[1], sketchy_data['0']))
    all_classes = list(berlin_classes.union(sketchy_classes))
    label_encoder = LabelEncoder()
    binary_classes = label_encoder.fit_transform(all_classes)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(binary_classes.reshape(-1, 1))
    for label in all_classes:
        numeric_label = label_encoder.transform(np.array([label]))
        class_map[label] = one_hot_encoder.transform(numeric_label.reshape(1,1)).reshape(-1)

    shapes_sequence = IndividualSequence(np.concatenate([shapes_data[str(i)] for i in list_of_folds]), 
                                                        shapes_targets, shapes_proportion, IMAGE_SIZE, shuffle = True)
    shapes_weights = {'classification': 0, 'mapping': 1, 'reconstruction': 1}
    
    additional_sequence = IndividualSequence(np.concatenate([additional_data[str(i)] for i in list_of_folds]), 
                                                            {None: 0}, additional_proportion, IMAGE_SIZE, shuffle = True)
    additional_weights = {'classification': 0, 'mapping': 0, 'reconstruction': 1}
    
    berlin_sequence = IndividualSequence(np.concatenate([berlin_data[str(i)] for i in list_of_folds]), 
                                                        class_map, berlin_proportion, IMAGE_SIZE, shuffle = True)
    berlin_weights = {'classification': 1, 'mapping': 0, 'reconstruction': 1}
    
    sketchy_sequence = IndividualSequence(np.concatenate([sketchy_data[str(i)] for i in list_of_folds]), 
                                                         class_map, sketchy_proportion, IMAGE_SIZE, shuffle = True)
    sketchy_weights = {'classification': 1, 'mapping': 0, 'reconstruction': 1}
    
    seqs = [shapes_sequence, additional_sequence, berlin_sequence, sketchy_sequence]
    weights = [shapes_weights, additional_weights, berlin_weights, sketchy_weights]
    
    data_sequence = OverallSequence(seqs, weights, space_dim, NUM_CLASSES)
    return data_sequence


test_fold = args.fold
val_fold = (test_fold - 1) % NUM_FOLDS
train_folds = [i for i in range(NUM_FOLDS) if i != test_fold and i != val_fold]

train_seq = get_data_sequence(train_folds)
val_seq = get_data_sequence([val_fold])
test_seq = get_data_sequence([test_fold])

# set up the model    
model = create_model()
callbacks = [tf.keras.callbacks.EarlyStopping()]
if args.walltime is not None:
    storage_path = '{0}_ep'.format(config_name)
    auto_restart = AutoRestart(filepath=storage_path, start_time=start_time, verbose = 0, walltime=args.walltime)
    callbacks.append(auto_restart)

if args.stopped_epoch > 0:
    model.load_weights(config_name + str(args.stopped_epoch) + '.hdf5')

# train it
history = model.fit_generator(generator = train_seq, steps_per_epoch = len(train_seq), epochs = EPOCHS, 
                              validation_data = val_seq, validation_steps = len(val_seq),
                              callbacks = callbacks, shuffle = True)


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
    evaluation_metrics = []
    evaluation_results = []

    # compute overall kendall correlation of bottleneck activation to dissimilarity ratings
    _, _, _, bottleneck_activation = model.predict(original_images)
    for distance_function in sorted(distance_functions.keys()):
        precomputed_distances = precompute_distances(bottleneck_activation, distance_function)
        kendall_fixed  = compute_correlations(precomputed_distances, target_dissimilarities, distance_function)['kendall']
        kendall_optimized = compute_correlations(precomputed_distances, target_dissimilarities, distance_function, 5, args.seed)['kendall']
        evaluation_metrics += ['kendall_{0}_fixed'.format(distance_function), 'kendall_{0}_optimized'.format(distance_function)]
        evaluation_results += [kendall_fixed, kendall_optimized]

    # compute standard evaluation metrics on the three tasks
    eval_train = model.evaluate_generator(train_seq, steps = len(train_seq))
    eval_val = model.evaluate_generator(val_seq, steps = len(val_seq))
    eval_test = model.evaluate_generator(test_seq, steps = len(test_seq))
    
    for evaluation, suffix in [(eval_train, '_train'), (eval_val, '_val'), (eval_test, '_test')]: 
        for metric_value, metric_name in zip(evaluation, model.metrics_names):
            evaluation_metrics.append(metric_name + suffix)
            evaluation_results.append(metric_value)
            print(metric_name + suffix, metric_value)
    
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