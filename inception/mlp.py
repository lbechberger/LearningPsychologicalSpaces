# -*- coding: utf-8 -*-
"""
Regression on the feature vectors with a multi-layer perceptron.

Created on Tue Jan 30 11:07:57 2018

@author: lbechberger
"""

import sys
import tensorflow as tf
import pickle
from math import sqrt, isnan
from random import shuffle
from configparser import RawConfigParser
import fcntl
import ast

layers = tf.contrib.layers

options = {}
options['features_file'] = 'features/images'
options['targets_file'] = 'features/targets-4d'
options['features_size'] = 2048
options['hidden_size'] = 128
options['space_size'] = 4
options['keep_prob'] = 0.8
options['alpha'] = 5.0             # influence of L2 loss
options['learning_rate'] = 0.003
options['epochs'] = '5'
options['batch_size'] = '64'

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")

def parse_range(key):
    value = options[key]
    parsed_value = ast.literal_eval(value)
    if isinstance(parsed_value, list):
        options[key] = parsed_value
    else:
        options[key] = [parsed_value]

if config.has_section(config_name):
    options['features_file'] = config.get(config_name, 'features_file')
    options['targets_file'] = config.get(config_name, 'targets_file')
    options['features_size'] = config.getint(config_name, 'features_size')
    options['hidden_size'] = config.getint(config_name, 'hidden_size')
    options['space_size'] = config.getint(config_name, 'space_size')
    options['keep_prob'] = config.getfloat(config_name, 'keep_prob')
    options['alpha'] = config.getfloat(config_name, 'alpha')
    options['learning_rate'] = config.getfloat(config_name, 'learning_rate')
    options['epochs'] = config.get(config_name, 'epochs')
    options['batch_size'] = config.get(config_name, 'batch_size')

parse_range('epochs')
parse_range('batch_size')

try:
    input_data = pickle.load(open(options['features_file'], 'rb'))
    targets_data = pickle.load(open(options['targets_file'], 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

real_targets = targets_data['targets']
shuffled_targets = targets_data['shuffled']

# defining the MLP regression network

tf_data = tf.placeholder(tf.float32, shape=[None, options['features_size']])
tf_labels = tf.placeholder(tf.float32, shape=[None, options['space_size']])

reg = layers.l2_regularizer(options['alpha'])
hidden = layers.fully_connected(tf_data, options['hidden_size'], weights_regularizer=reg, biases_regularizer=reg)
dropout = tf.nn.dropout(hidden, options['keep_prob'])
prediction = layers.fully_connected(dropout, options['space_size'], weights_regularizer=reg, biases_regularizer=reg, activation_fn=None)
mse = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - tf_labels), axis=1))  

global_step = tf.Variable(0)
optimizer = tf.train.GradientDescentOptimizer(options['learning_rate']).minimize(mse, global_step = global_step)

squared_train_errors = {'real': {}, 'shuffled': {}}
squared_test_errors = {'real': {}, 'shuffled': {}}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    for test_image in input_data.keys():
        print("processing test image {0}".format(test_image))
        train_image_names = [img_name for img_name in input_data.keys() if img_name != test_image]
        
        features_train = []
        real_labels_train = []
        shuffled_labels_train = []
        for img_name in train_image_names:
            features_train += input_data[img_name]
            real_labels_train += [real_targets[img_name]]*len(input_data[img_name])
            shuffled_labels_train += [shuffled_targets[img_name]]*len(input_data[img_name])
        
        zipped = list(zip(features_train, real_labels_train, shuffled_labels_train))
        shuffle(zipped)
        features_train = list(map(lambda x: x[0], zipped))
        labels_train = {}
        labels_train['real'] = list(map(lambda x: x[1], zipped))
        labels_train['shuffled'] = list(map(lambda x: x[2], zipped))
        
        features_test = input_data[test_image]
        labels_test = {}
        labels_test['real'] = [real_targets[img_name]]*len(input_data[test_image])
        labels_test['shuffled'] = [shuffled_targets[img_name]]*len(input_data[test_image])
        
        for batch_size in options['batch_size']:
            squared_train_errors['real'][batch_size] = {}
            squared_train_errors['shuffled'][batch_size] = {}
            squared_test_errors['real'][batch_size] = {}
            squared_test_errors['shuffled'][batch_size] = {}
            
            num_steps = {}
            max_num_steps = 0
            for epoch in options['epochs']:
                steps = int( (epoch * len(features_train)) / batch_size )
                num_steps[steps] = epoch
                max_num_steps = max(max_num_steps, steps)
                squared_train_errors['real'][batch_size][epoch] = []
                squared_train_errors['shuffled'][batch_size][epoch] = []
                squared_test_errors['real'][batch_size][epoch] = []
                squared_test_errors['shuffled'][batch_size][epoch] = []

            def train_regression(session, target_type):
                tf.global_variables_initializer().run()
                for step in range(max_num_steps):
                    offset = (step * batch_size) % (len(features_train) - batch_size)
                    batch_data = features_train[offset:(offset + batch_size)]
                    batch_labels = labels_train[target_type][offset:(offset + batch_size)]
                    
                    feed_dict = {tf_data : batch_data, tf_labels : batch_labels}
                    _, l = session.run([optimizer, mse], feed_dict = feed_dict)  
                    if isnan(l):
                        print("Loss NaN in step {0}!".format(step))
                        break
                    if (step + 1) in num_steps.keys():
                        epoch = num_steps[step + 1]
                        local_train_mse = session.run(mse, feed_dict = {tf_data : features_train, tf_labels : labels_train[target_type]})
                        squared_train_errors[target_type][batch_size][epoch].append(local_train_mse)
                        local_test_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : labels_test[target_type]})
                        squared_test_errors[target_type][batch_size][epoch].append(local_test_mse)

            # first train real
            train_regression(session, 'real')   
            # now train shuffled
            train_regression(session, 'shuffled')   
            

def rmse(mse_list):
    average_mse = (1.0 * sum(mse_list)) / len(mse_list)
    return sqrt(average_mse)

for batch_size in options['batch_size']:
    for epoch in options['epochs']:
        for label_type in ['real', 'shuffled']:
            combination_label = "{0}-ba{1}-ep{2}-{3}".format(config_name, batch_size, epoch, label_type)
            train_rmse = rmse(squared_train_errors[label_type][batch_size][epoch])
            test_rmse = rmse(squared_test_errors[label_type][batch_size][epoch])
            print("Train RMSE for {0}: {1}".format(combination_label, train_rmse))
            print("Test RMSE for {0}: {1}".format(combination_label, test_rmse))
            
            with open("mlp/{0}".format(combination_label), 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write("{0},{1}\n".format(train_rmse, test_rmse))
                fcntl.flock(f, fcntl.LOCK_UN)
