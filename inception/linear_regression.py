# -*- coding: utf-8 -*-
"""
Simple linear regression on the feature vectors.

Created on Tue Jan 30 11:07:57 2018

@author: lbechberger
"""

import os
import tensorflow as tf
import numpy as np
import pickle
from math import sqrt

flags = tf.flags
flags.DEFINE_string('features_dir', 'features', 'Directory where the feature vectors reside.')
flags.DEFINE_integer('features_size', 2048, 'Size of the feature vector.')
flags.DEFINE_integer('space_size', 4, 'Size of the psychological space.')
flags.DEFINE_integer('num_steps', 200, 'Number of optimization steps.')
flags.DEFINE_integer('num_repetitions', 5, 'Number of repetitions for each fold.')
flags.DEFINE_float('keep_prob', 0.8, 'Keep probability for dropout.')
flags.DEFINE_float('alpha', 5.0, 'Influence of L2 loss.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')

FLAGS = flags.FLAGS

features = np.array(pickle.load(open(os.path.join(FLAGS.features_dir, 'features'))))
labels = np.array(pickle.load(open(os.path.join(FLAGS.features_dir, 'labels'))))

weights = tf.Variable(tf.truncated_normal([FLAGS.features_size,FLAGS.space_size]))
bias = tf.Variable(tf.truncated_normal([FLAGS.space_size]))
tf_data = tf.placeholder(tf.float32, shape=[None, FLAGS.features_size])
tf_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.space_size])

dropout = tf.nn.dropout(tf_data, FLAGS.keep_prob)
prediction = tf.matmul(dropout, weights) + bias
mse = tf.reduce_mean(tf.square(prediction - tf_labels))    

global_step = tf.Variable(0)
loss = mse + FLAGS.alpha * tf.nn.l2_loss(weights)
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step = global_step)

squared_errors = []
all_indices = range(len(features))
for test_index in all_indices:
    train_index = [i for i in all_indices if i != test_index]
    features_train = np.array([features[i] for i in train_index])
    labels_train = np.array([labels[i] for i in train_index])
    features_test = np.array([features[test_index]])
    labels_test = np.array([labels[test_index]])
    
    for repetition in range(FLAGS.num_repetitions):
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for step in range(FLAGS.num_steps):
                feed_dict = {tf_data : features_train, tf_labels : labels_train}
                _, l = session.run([optimizer, loss], feed_dict = feed_dict)
            
            local_mse = session.run(mse, feed_dict = {tf_data : features_test, tf_labels : labels_test})
            squared_errors.append(local_mse)

overall_mse = sum(squared_errors) / len(squared_errors)
rmse = sqrt(overall_mse)
print("Overall RMSE: {0}".format(rmse))


            
        