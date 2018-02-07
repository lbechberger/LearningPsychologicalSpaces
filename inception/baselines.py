# -*- coding: utf-8 -*-
"""
Two baselines: always predict 0 and always predict a random point.

Created on Wed Feb  7 10:56:46 2018

@author: lbechberger
"""


import os, sys
import tensorflow as tf
import pickle
from math import sqrt
from random import shuffle
from configparser import RawConfigParser
import fcntl

options = {}
options['features_file'] = 'features/images'
options['targets_file'] = 'features/targets-4d'
options['space_size'] = 4

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("regression.cfg")

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

# defining the zero baseline network
zero_output = tf.zeros(shape=(1, options['space_size']))
zero_mse = tf.reduce_mean(tf.square(zero_output - tf_labels))

# defining the random baseline network
normal_dist = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(shape=(1,options['space_size'])), [0.4]*options['space_size'])
random_output = normal_dist.sample()
random_mse = tf.reduce_mean(tf.square(random_output - tf_labels))    

real_squared_train_errors_zero = []
real_squared_test_errors_zero = []
real_squared_train_errors_random = []
real_squared_test_errors_random = []

shuffled_squared_train_errors_zero = []
shuffled_squared_test_errors_zero = []
shuffled_squared_train_errors_random = []
shuffled_squared_test_errors_random = []

for test_image in input_data.keys():
    
    train_image_names = [img_name for img_name in input_data.keys() if img_name != test_image]
    
    real_labels_train = []
    shuffled_labels_train = []
    for img_name in train_image_names:
        real_labels_train += [real_targets[img_name]]*len(input_data[img_name])
        shuffled_labels_train += [shuffled_targets[img_name]]*len(input_data[img_name])
    
    real_labels_test = [real_targets[test_image]]*len(input_data[test_image])
    shuffled_labels_test = [shuffled_targets[test_image]]*len(input_data[test_image])
    
    # first evaluate the real targets
    with tf.Session() as session:
        tf.global_variables_initializer().run()
             
        local_train_mse = session.run(zero_mse, feed_dict = {tf_labels : real_labels_train})
        real_squared_train_errors_zero.append(local_train_mse) 
        local_test_mse = session.run(zero_mse, feed_dict = {tf_labels : real_labels_test})
        real_squared_test_errors_zero.append(local_test_mse)
        
        local_train_mse = session.run(random_mse, feed_dict = {tf_labels : real_labels_train})
        real_squared_train_errors_random.append(local_train_mse) 
        local_test_mse = session.run(random_mse, feed_dict = {tf_labels : real_labels_test})
        real_squared_test_errors_random.append(local_test_mse)
    
    # and now the shuffled targets
    with tf.Session() as session:
        tf.global_variables_initializer().run()
             
        local_train_mse = session.run(zero_mse, feed_dict = {tf_labels : shuffled_labels_train})
        shuffled_squared_train_errors_zero.append(local_train_mse) 
        local_test_mse = session.run(zero_mse, feed_dict = {tf_labels : shuffled_labels_test})
        shuffled_squared_test_errors_zero.append(local_test_mse)
        
        local_train_mse = session.run(random_mse, feed_dict = {tf_labels : shuffled_labels_train})
        shuffled_squared_train_errors_random.append(local_train_mse) 
        local_test_mse = session.run(random_mse, feed_dict = {tf_labels : shuffled_labels_test})
        shuffled_squared_test_errors_random.append(local_test_mse)


# real
real_train_mse_zero = sum(real_squared_train_errors_zero) / len(real_squared_train_errors_zero)
real_train_rmse_zero = sqrt(real_train_mse_zero)
print("Overall RMSE on training set for 'zero' baseline on real targets: {0}".format(real_train_rmse_zero))

real_test_mse_zero = sum(real_squared_test_errors_zero) / len(real_squared_test_errors_zero)
real_test_rmse_zero = sqrt(real_test_mse_zero)
print("Overall RMSE on test set for 'zero' baseline on real targets: {0}".format(real_test_rmse_zero))

real_train_mse_random = sum(real_squared_train_errors_random) / len(real_squared_train_errors_random)
real_train_rmse_random = sqrt(real_train_mse_random)
print("Overall RMSE on training set for 'random' baseline on real targets: {0}".format(real_train_rmse_random))

real_test_mse_random = sum(real_squared_test_errors_random) / len(real_squared_test_errors_random)
real_test_rmse_random = sqrt(real_test_mse_random)
print("Overall RMSE on test set for 'random' baseline on real targets: {0}".format(real_test_rmse_random))

# shuffled
shuffled_train_mse_zero = sum(shuffled_squared_train_errors_zero) / len(shuffled_squared_train_errors_zero)
shuffled_train_rmse_zero = sqrt(shuffled_train_mse_zero)
print("Overall RMSE on training set for 'zero' baseline on shuffled targets: {0}".format(shuffled_train_rmse_zero))

shuffled_test_mse_zero = sum(shuffled_squared_test_errors_zero) / len(shuffled_squared_test_errors_zero)
shuffled_test_rmse_zero = sqrt(shuffled_test_mse_zero)
print("Overall RMSE on test set for 'zero' baseline on shuffled targets: {0}".format(shuffled_test_rmse_zero))

shuffled_train_mse_random = sum(shuffled_squared_train_errors_random) / len(shuffled_squared_train_errors_random)
shuffled_train_rmse_random = sqrt(shuffled_train_mse_random)
print("Overall RMSE on training set for 'random' baseline on shuffled targets: {0}".format(shuffled_train_rmse_random))

shuffled_test_mse_random = sum(shuffled_squared_test_errors_random) / len(shuffled_squared_test_errors_random)
shuffled_test_rmse_random = sqrt(shuffled_test_mse_random)
print("Overall RMSE on test set for 'random' baseline on shuffled targets: {0}".format(shuffled_test_rmse_random))


with open("regression/baseline_zero_{0}-real".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(real_train_rmse_zero, real_test_rmse_zero))
    fcntl.flock(f, fcntl.LOCK_UN)

with open("regression/baseline_random_{0}-real".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(real_train_rmse_random, real_test_rmse_random))
    fcntl.flock(f, fcntl.LOCK_UN)

with open("regression/baseline_zero_{0}-shuffled".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(shuffled_train_rmse_zero, shuffled_test_rmse_zero))
    fcntl.flock(f, fcntl.LOCK_UN)

with open("regression/baseline_random_{0}-shuffled".format(config_name), 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write("{0},{1}\n".format(shuffled_train_rmse_random, shuffled_test_rmse_random))
    fcntl.flock(f, fcntl.LOCK_UN)