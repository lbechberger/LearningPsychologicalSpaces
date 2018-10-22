# -*- coding: utf-8 -*-
"""
Two baselines: always predict 0 and always predict a random point.

Created on Wed Feb  7 10:56:46 2018

@author: lbechberger
"""


import sys
import tensorflow as tf
import pickle
from math import sqrt
from configparser import RawConfigParser
import fcntl

options = {}
options['features_file'] = 'features/images'
options['targets_file'] = 'features/targets-4d'
options['space_size'] = 4

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")

if config.has_section(config_name):
    options['features_file'] = config.get(config_name, 'features_file')
    options['targets_file'] = config.get(config_name, 'targets_file')
    options['space_size'] = config.getint(config_name, 'space_size')

try:
    input_data = pickle.load(open(options['features_file'], 'rb'))
    targets_data = pickle.load(open(options['targets_file'], 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

real_targets = targets_data['targets']
shuffled_targets = targets_data['shuffled']

tf_labels = tf.placeholder(tf.float32, shape=[None, options['space_size']])
tf_labels_in = tf.placeholder(tf.float32, shape=[None, options['space_size']])

# defining the zero baseline network
zero_output = tf.zeros(shape=(1, options['space_size']))
zero_mse = tf.reduce_mean(tf.reduce_sum(tf.square(zero_output - tf_labels), axis=1))

# defining the mean baseline network
mean_output = tf.reduce_mean(tf_labels_in, axis=0)
mean_mse = tf.reduce_mean(tf.reduce_sum(tf.square(mean_output - tf_labels), axis=1))

# defining the distribution baseline network
mu = mean_output
difference = tf.expand_dims(tf_labels - mu, axis = 1)
multiplication = tf.matmul(difference, difference, transpose_a=True)
var = tf.reduce_mean(multiplication, axis=0)
var = var + 1e-7 * tf.eye(options['space_size']) # ensure positive definiteness
dist = tf.contrib.distributions.MultivariateNormalFullCovariance(mu, var)
dist_output = dist.sample()
dist_mse = tf.reduce_mean(tf.reduce_sum(tf.square(dist_output - tf_labels), axis=1))

# defining the random draw baseline network
indices = tf.random_uniform([tf.shape(tf_labels)[0]], minval=0, maxval=tf.shape(tf_labels_in)[0], dtype=tf.int32)
draw_output = tf.gather(tf_labels_in, indices)
draw_mse = tf.reduce_mean(tf.reduce_sum(tf.square(draw_output - tf_labels), axis=1))


squared_train_errors = {'real': {}, 'shuffled': {}}
squared_test_errors = {'real': {}, 'shuffled': {}}

baseline_methods = ['zero', 'mean', 'dist', 'draw']
for method in baseline_methods:
    for labels in ['real', 'shuffled']:
        for data_set in [squared_train_errors, squared_test_errors]:
            data_set[labels][method] = []


for test_image in input_data.keys():
    
    train_image_names = [img_name for img_name in input_data.keys() if img_name != test_image]
    
    real_labels_train = []
    shuffled_labels_train = []
    for img_name in train_image_names:
        real_labels_train += [real_targets[img_name]]*len(input_data[img_name])
        shuffled_labels_train += [shuffled_targets[img_name]]*len(input_data[img_name])
    
    real_labels_test = [real_targets[test_image]]*len(input_data[test_image])
    shuffled_labels_test = [shuffled_targets[test_image]]*len(input_data[test_image])

    labels_train = {}
    labels_train['real'] = real_labels_train
    labels_train['shuffled'] = shuffled_labels_train
    
    labels_test = {}
    labels_test['real'] = real_labels_test
    labels_test['shuffled'] = shuffled_labels_test

    
    with tf.Session() as session:
        
        def evaluate_baselines(session, target_type):
            tf.global_variables_initializer().run()
            feed_dict_train = {tf_labels: labels_train[target_type], tf_labels_in: labels_train[target_type]}
            feed_dict_test = {tf_labels: labels_test[target_type], tf_labels_in: labels_train[target_type]}
            train_zero, train_mean, train_dist, train_draw = session.run([zero_mse, mean_mse, dist_mse, draw_mse], feed_dict = feed_dict_train)
            test_zero, test_mean, test_dist, test_draw = session.run([zero_mse, mean_mse, dist_mse, draw_mse], feed_dict = feed_dict_test)
            squared_train_errors[target_type]['zero'].append(train_zero)
            squared_train_errors[target_type]['mean'].append(train_mean)
            squared_train_errors[target_type]['dist'].append(train_dist)
            squared_train_errors[target_type]['draw'].append(train_draw)
            squared_test_errors[target_type]['zero'].append(test_zero)
            squared_test_errors[target_type]['mean'].append(test_mean)
            squared_test_errors[target_type]['dist'].append(test_dist)
            squared_test_errors[target_type]['draw'].append(test_draw)
            
        evaluate_baselines(session, 'real')
        evaluate_baselines(session, 'shuffled')


def rmse(mse_list):
    average_mse = (1.0 * sum(mse_list)) / len(mse_list)
    return sqrt(average_mse)

for labels in ['real', 'shuffled']:
    for method in baseline_methods:
        combination_label = "baseline-{0}-{1}".format(method,labels)
        train_rmse = rmse(squared_train_errors[labels][method])
        test_rmse = rmse(squared_test_errors[labels][method])
        print("Train RMSE for {0}: {1}".format(combination_label, train_rmse))
        print("Test RMSE for {0}: {1}".format(combination_label, test_rmse))
        
        with open("regression/{0}".format(combination_label), 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("{0},{1}\n".format(train_rmse, test_rmse))
            fcntl.flock(f, fcntl.LOCK_UN)
