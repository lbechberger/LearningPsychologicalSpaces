# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:05:26 2018

@author: lbechberger
"""

import os, sys
import tensorflow as tf
import numpy as np
import pickle
from math import sqrt

flags = tf.flags
flags.DEFINE_string('features_dir', 'features', 'Directory where the feature vectors reside.')
flags.DEFINE_integer('features_size', 2048, 'Size of the feature vector.')
flags.DEFINE_integer('target_size', 1024, 'Size of the hidden layer.')
flags.DEFINE_integer('num_steps', 2000, 'Number of optimization steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size used during training.')
flags.DEFINE_float('keep_prob', 0.8, 'Keep probability for dropout.')
flags.DEFINE_float('alpha', 0.5, 'Influence of L2 loss.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')

FLAGS = flags.FLAGS

try:
    all_data = pickle.load(open(os.path.join(FLAGS.features_dir, 'features'), 'rb'))
except Exception:
    print("Cannot read input data. Aborting.")
    sys.exit(0)

features = []
for (data, _, _) in all_data.values():
    features += data

# define neural network (i.e., autoencoder)
weights_encoder = tf.Variable(tf.truncated_normal([FLAGS.features_size,FLAGS.target_size]))
weights_decoder = tf.Variable(tf.truncated_normal([FLAGS.target_size,FLAGS.features_size]))
bias_encoder = tf.Variable(tf.truncated_normal([FLAGS.target_size]))
bias_decoder = tf.Variable(tf.truncated_normal([FLAGS.features_size]))
tf_data = tf.placeholder(tf.float32, shape=[None, FLAGS.features_size])

hidden = tf.nn.relu(tf.matmul(tf_data, weights_encoder) + bias_encoder)
dropout = tf.nn.dropout(hidden, FLAGS.keep_prob)
prediction = tf.matmul(dropout, weights_decoder) + bias_decoder
mse = tf.reduce_mean(tf.square(prediction - tf_data))    

global_step = tf.Variable(0)
loss = mse + FLAGS.alpha * (tf.nn.l2_loss(weights_encoder) + tf.nn.l2_loss(bias_encoder) + tf.nn.l2_loss(weights_decoder) + tf.nn.l2_loss(bias_decoder))
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step = global_step)

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for step in range(FLAGS.num_steps):
        offset = (step * FLAGS.batch_size) % (len(features) - FLAGS.batch_size)
        batch_data = features[offset:(offset + FLAGS.batch_size)]
        feed_dict = {tf_data : batch_data}
        _, l = session.run([optimizer, loss], feed_dict = feed_dict)
        if step % 100 == 0:
            print(step, l)
    
    local_mse = session.run(mse, feed_dict = {tf_data : features})
    print("Overall RMSE: {0}".format(sqrt(local_mse)))
    
    compressed_representation = session.run(hidden, feed_dict = {tf_data : features})
    pickle.dump(np.squeeze(compressed_representation), open(os.path.join(FLAGS.features_dir, 'features_compressed'), 'wb'))