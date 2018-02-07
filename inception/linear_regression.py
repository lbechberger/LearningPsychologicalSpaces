# -*- coding: utf-8 -*-
"""
Simple linear regression on the feature vectors.

Created on Tue Jan 30 11:07:57 2018

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
options['features_size'] = 2048
options['space_size'] = 4
options['num_steps'] = 200
options['batch_size'] = 64
options['keep_prob'] = 0.8
options['alpha'] = 5.0             # influence of L2 loss
options['learning_rate'] = 0.01

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("regression.cfg")

if config.has_section(config_name):
    options['features_dir'] = config.get(config_name, 'features_dir')
    options['features_size'] = config.getint(config_name, 'features_size')
    options['space_size'] = config.getint(config_name, 'space_size')
    options['num_steps'] = config.getint(config_name, 'num_steps')
    options['batch_size'] = config.getint(config_name, 'batch_size')
    options['keep_prob'] = config.getfloat(config_name, 'keep_prob')
    options['alpha'] = config.getfloat(config_name, 'alpha')
    options['learning_rate'] = config.getfloat(config_name, 'learning_rate')

try:
    all_data = pickle.load(open(os.path.join(options['features_dir'], 'all'), 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

# defining the linear regression network
weights = tf.Variable(tf.truncated_normal([options['features_size'],options['space_size']]))
bias = tf.Variable(tf.truncated_normal([options['space_size']]))
tf_data = tf.placeholder(tf.float32, shape=[None, options['features_size']])
tf_labels = tf.placeholder(tf.float32, shape=[None, options['space_size']])

dropout = tf.nn.dropout(tf_data, options['keep_prob'])
prediction = tf.matmul(dropout, weights) + bias
mse = tf.reduce_mean(tf.square(prediction - tf_labels))    

global_step = tf.Variable(0)
loss = mse + options['alpha'] * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
optimizer = tf.train.GradientDescentOptimizer(options['learning_rate']).minimize(loss, global_step = global_step)

squared_train_errors = []
squared_test_errors = []
predictions = []
all_weights = []
all_biases = []

for test_image in all_data.keys():
    
    train_image_names = [img_name for img_name in all_data.keys() if img_name != test_image]
    
    features_train = []
    labels_train = []
    for img_name in train_image_names:
        augmented, target, original = all_data[img_name]
        features_train += augmented
        labels_train += [target]*len(augmented)
    
    zipped = list(zip(features_train, labels_train))
    shuffle(zipped)
    features_train = list(map(lambda x: x[0], zipped))
    labels_train = list(map(lambda x: x[1], zipped))
    
    augmented, target, original = all_data[test_image]
    features_test = augmented
    labels_test = [target]*len(augmented)
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for step in range(options['num_steps']):
            offset = (step * options['batch_size']) % (len(labels_train) - options['batch_size'])
            batch_data = features_train[offset:(offset + options['batch_size'])]
            batch_labels = labels_train[offset:(offset + options['batch_size'])]
            print(len(batch_data))
    
            feed_dict = {tf_data : batch_data, tf_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict = feed_dict)    
            
        local_test_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : labels_test})
        preds = session.run(prediction, feed_dict = {tf_data : features_test, tf_labels : labels_test})
        predictions.append(preds)
        all_weights.append(session.run(weights))
        all_biases.append(session.run(bias))
        squared_test_errors.append(local_test_mse)
        local_train_mse = session.run(mse, feed_dict = {tf_data : features_train, tf_labels : labels_train})
        squared_train_errors.append(local_train_mse)

print("Individual predictions: {0}".format(predictions))
print("Weight matrices: {0}".format(all_weights))
print("Bias terms: {0}".format(all_biases))
overall_train_mse = sum(squared_train_errors) / len(squared_train_errors)
train_rmse = sqrt(overall_train_mse)
print("batch-wise training results: {0}".format(squared_train_errors))
print("Overall RMSE on training set: {0}".format(train_rmse))

overall_test_mse = sum(squared_test_errors) / len(squared_test_errors)
test_rmse = sqrt(overall_test_mse)
print("batch-wise test results: {0}".format(squared_test_errors))
print("Overall RMSE on test set: {0}".format(test_rmse))

with open("regression/{0}".format(config_name), 'a') as f:
    f.write("{0},{1}\n".format(train_rmse, test_rmse))