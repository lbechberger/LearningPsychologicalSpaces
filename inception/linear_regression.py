# -*- coding: utf-8 -*-
"""
Simple linear regression on the feature vectors.

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

options = {}
options['features_file'] = 'features/images'
options['targets_file'] = 'features/targets-4d'
options['features_size'] = 2048
options['space_size'] = 4
options['num_steps'] = 2000
options['batch_size'] = 64
options['keep_prob'] = 0.8
options['alpha'] = 5.0             # influence of L2 loss
options['learning_rate'] = 0.003

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")

if config.has_section(config_name):
    options['features_file'] = config.get(config_name, 'features_file')
    options['targets_file'] = config.get(config_name, 'targets_file')
    options['features_size'] = config.getint(config_name, 'features_size')
    options['space_size'] = config.getint(config_name, 'space_size')
    options['num_steps'] = config.getint(config_name, 'num_steps')
    options['batch_size'] = config.getint(config_name, 'batch_size')
    options['keep_prob'] = config.getfloat(config_name, 'keep_prob')
    options['alpha'] = config.getfloat(config_name, 'alpha')
    options['learning_rate'] = config.getfloat(config_name, 'learning_rate')

try:
    input_data = pickle.load(open(options['features_file'], 'rb'))
    targets_data = pickle.load(open(options['targets_file'], 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

real_targets = targets_data['targets']
shuffled_targets = targets_data['shuffled']

# defining the linear regression network
weights = tf.Variable(tf.truncated_normal([options['features_size'],options['space_size']]))
bias = tf.Variable(tf.truncated_normal([options['space_size']]))
tf_data = tf.placeholder(tf.float32, shape=[None, options['features_size']])
tf_labels = tf.placeholder(tf.float32, shape=[None, options['space_size']])

dropout = tf.nn.dropout(tf_data, options['keep_prob'])
prediction = tf.matmul(dropout, weights) + bias
mse = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - tf_labels), axis=1))  

global_step = tf.Variable(0)
loss = mse + options['alpha'] * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
optimizer = tf.train.GradientDescentOptimizer(options['learning_rate']).minimize(loss, global_step = global_step)

real_squared_train_errors = []
real_squared_test_errors = []
shuffled_squared_train_errors = []
shuffled_squared_test_errors = []

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
        real_labels_train = list(map(lambda x: x[1], zipped))
        shuffled_labels_train = list(map(lambda x: x[2], zipped))
        
        features_test = input_data[test_image]
        real_labels_test = [real_targets[img_name]]*len(input_data[test_image])
        shuffled_labels_test = [shuffled_targets[img_name]]*len(input_data[test_image])
        
        # first train real
        tf.global_variables_initializer().run()
        for step in range(options['num_steps']):
            offset = (step * options['batch_size']) % (len(features_train) - options['batch_size'])
            batch_data = features_train[offset:(offset + options['batch_size'])]
            batch_labels = real_labels_train[offset:(offset + options['batch_size'])]
            
            feed_dict = {tf_data : batch_data, tf_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict = feed_dict)  
            #print("Loss in step {0}: {1}".format(step, l))
            if isnan(l):
                print("Loss NaN in step {0}!".format(step))
                break
            
        local_train_mse = session.run(mse, feed_dict = {tf_data : features_train, tf_labels : real_labels_train})
        real_squared_train_errors.append(local_train_mse)
        local_test_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : real_labels_test})
        real_squared_test_errors.append(local_test_mse)
        
        # now train shuffled
        tf.global_variables_initializer().run()
        for step in range(options['num_steps']):
            offset = (step * options['batch_size']) % (len(features_train) - options['batch_size'])
            batch_data = features_train[offset:(offset + options['batch_size'])]
            batch_labels = shuffled_labels_train[offset:(offset + options['batch_size'])]
            
            feed_dict = {tf_data : batch_data, tf_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict = feed_dict)    
            if isnan(l):
                print("Loss NaN in step {0}!".format(step))
                break
            
        local_train_mse = session.run(mse, feed_dict = {tf_data : features_train, tf_labels : shuffled_labels_train})
        shuffled_squared_train_errors.append(local_train_mse)
        local_test_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : shuffled_labels_test})
        shuffled_squared_test_errors.append(local_test_mse)
        
real_train_mse = sum(real_squared_train_errors) / len(real_squared_train_errors)
real_train_rmse = sqrt(real_train_mse)
print("Overall RMSE on training set with real targets: {0}".format(real_train_rmse))

real_test_mse = sum(real_squared_test_errors) / len(real_squared_test_errors)
real_test_rmse = sqrt(real_test_mse)
print("Overall RMSE on test set with real targets: {0}".format(real_test_rmse))

shuffled_train_mse = sum(shuffled_squared_train_errors) / len(shuffled_squared_train_errors)
shuffled_train_rmse = sqrt(shuffled_train_mse)
print("Overall RMSE on training set with shuffled targets: {0}".format(shuffled_train_rmse))

shuffled_test_mse = sum(shuffled_squared_test_errors) / len(shuffled_squared_test_errors)
shuffled_test_rmse = sqrt(shuffled_test_mse)
print("Overall RMSE on test set with shuffled targets: {0}".format(shuffled_test_rmse))

with open("regression/{0}-real".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(real_train_rmse, real_test_rmse))
    fcntl.flock(f, fcntl.LOCK_UN)
    
with open("regression/{0}-shuffled".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(shuffled_train_rmse, shuffled_test_rmse))
    fcntl.flock(f, fcntl.LOCK_UN)