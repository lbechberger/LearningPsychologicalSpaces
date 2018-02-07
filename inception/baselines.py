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

options = {}
options['features_dir'] = 'features/features'
options['space_size'] = 4

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("regression.cfg")

if config.has_section(config_name):
    options['features_dir'] = config.get(config_name, 'features_dir')
    options['space_size'] = config.getint(config_name, 'space_size')

try:
    all_data = pickle.load(open(os.path.join(options['features_dir'], 'all'), 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)


tf_labels = tf.placeholder(tf.float32, shape=[None, options['space_size']])

# defining the zero baseline network
zero_output = tf.zeros(shape=tf_labels.shape)
zero_mse = tf.reduce_mean(tf.square(zero_output - tf_labels))

# defining the random baseline network
normal_dist = tf.contrib.distribtutions.MultivariateNormalDiag(tf.zeros(shape=(1,options['space_size']), tf.constant(0.4, shape=(options['space_size']))))
random_output = normal_dist.sample(tf_labels.shape[0])
random_mse = tf.reduce_mean(tf.square(random_mse - tf_labels))    

squared_train_errors_zero = []
squared_test_errors_zero = []
squared_train_errors_random = []
squared_test_errors_random = []

for test_image in all_data.keys():
    
    train_image_names = [img_name for img_name in all_data.keys() if img_name != test_image]
    
    labels_train = []
    for img_name in train_image_names:
        augmented, target, original = all_data[img_name]
        labels_train += [target]*len(augmented)
    
    shuffle(labels_train)
    
    augmented, target, original = all_data[test_image]
    labels_test = [target]*len(augmented)
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
             
        local_train_mse = session.run(zero_mse, feed_dict = {tf_labels : labels_train})
        squared_train_errors_zero.append(local_train_mse) 
        local_test_mse = session.run(zero_mse, feed_dict = {tf_labels : labels_test})
        squared_test_errors_zero.append(local_test_mse)
        
        local_train_mse = session.run(random_mse, feed_dict = {tf_labels : labels_train})
        squared_train_errors_random.append(local_train_mse) 
        local_test_mse = session.run(random_mse, feed_dict = {tf_labels : labels_test})
        squared_test_errors_random.append(local_test_mse)


overall_train_mse_zero = sum(squared_train_errors_zero) / len(squared_train_errors_zero)
train_rmse_zero = sqrt(overall_train_mse_zero)
print("Overall RMSE on training set for 'zero' baseline: {0}".format(train_rmse_zero))

overall_test_mse_zero = sum(squared_test_errors_zero) / len(squared_test_errors_zero)
test_rmse_zero = sqrt(overall_test_mse_zero)
print("Overall RMSE on test set for 'zero' baseline: {0}".format(test_rmse_zero))

overall_train_mse_random = sum(squared_train_errors_random) / len(squared_train_errors_random)
train_rmse_random = sqrt(overall_train_mse_random)
print("Overall RMSE on training set for 'random' baseline: {0}".format(train_rmse_random))

overall_test_mse_random = sum(squared_test_errors_random) / len(squared_test_errors_random)
test_rmse_random = sqrt(overall_test_mse_random)
print("Overall RMSE on test set for 'random' baseline: {0}".format(test_rmse_random))
