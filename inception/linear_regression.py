# -*- coding: utf-8 -*-
"""
Simple linear regression on the feature vectors.

Created on Tue Jan 30 11:07:57 2018

@author: lbechberger
"""

import os, sys
import tensorflow as tf
import numpy as np
import pickle
from math import sqrt
from random import shuffle

flags = tf.flags
flags.DEFINE_string('features_dir', 'features', 'Directory where the feature vectors reside.')
flags.DEFINE_integer('features_size', 2048, 'Size of the feature vector.')
flags.DEFINE_integer('space_size', 4, 'Size of the psychological space.')
flags.DEFINE_integer('num_steps', 200, 'Number of optimization steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size used during training.')
flags.DEFINE_float('keep_prob', 0.8, 'Keep probability for dropout.')
flags.DEFINE_float('alpha', 5.0, 'Influence of L2 loss.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')

FLAGS = flags.FLAGS

try:
    all_data = pickle.load(open(os.path.join(FLAGS.features_dir, 'features'), 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

# defining the linear regression network
weights = tf.Variable(tf.truncated_normal([FLAGS.features_size,FLAGS.space_size]))
bias = tf.Variable(tf.truncated_normal([FLAGS.space_size]))
tf_data = tf.placeholder(tf.float32, shape=[None, FLAGS.features_size])
tf_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.space_size])

dropout = tf.nn.dropout(tf_data, FLAGS.keep_prob)
prediction = tf.matmul(dropout, weights) + bias
mse = tf.reduce_mean(tf.square(prediction - tf_labels))    

global_step = tf.Variable(0)
loss = mse + FLAGS.alpha * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step = global_step)

squared_test_errors = []
squared_train_errors = []

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
        for step in range(FLAGS.num_steps):
            offset = (step * FLAGS.batch_size) % (len(labels_train) - FLAGS.batch_size)
            batch_data = features_train[offset:(offset + FLAGS.batch_size)]
            batch_labels = labels_train[offset:(offset + FLAGS.batch_size)]
    
            feed_dict = {tf_data : batch_data, tf_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict = feed_dict)
            
            if step%100 == 0:
                print("Minibatch loss at step {0}: {1}".format(step, l))
        
        local_test_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : labels_test})
        squared_test_errors.append(local_test_mse)
        local_train_mse = session.run(mse, feed_dict = {tf_data : features_train, tf_labels : labels_train})
        squared_train_errors.append(local_train_mse)

overall_test_mse = sum(squared_test_errors) / len(squared_test_errors)
test_rmse = sqrt(overall_test_mse)
print("Overall RMSE on test set: {0}".format(test_rmse))
overall_train_mse = sum(squared_train_errors) / len(squared_train_errors)
train_rmse = sqrt(overall_train_mse)
print("Overall RMSE on training set: {0}".format(train_rmse))