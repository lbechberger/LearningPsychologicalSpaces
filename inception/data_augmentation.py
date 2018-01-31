# -*- coding: utf-8 -*-
"""
Script for creating an augmented data set.

Inspired by https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

Created on Tue Jan 30 10:49:25 2018

@author: lbechberger
"""

import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle

flags = tf.flags
flags.DEFINE_string('model_dir', '/tmp/imagenet/', 'Directory where the pretrained network resides.')
flags.DEFINE_string('images_dir', '../images/', 'Location of data.')
flags.DEFINE_string('output_dir', 'features', 'Where to store the feature vectors.')
flags.DEFINE_string('mapping_file', 'mapping.csv', 'CSV file mapping image names to target vectors.')
flags.DEFINE_integer('n_dim', 4, 'Number of target dimensions.')

FLAGS = flags.FLAGS

def augment_image(base_image, num_samples=1000):

    #TODO do some real augmenting    
    
    encoder = tf.image.encode_jpeg(base_image)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        encoded_image = session.run(encoder)
    return np.array([encoded_image])

image_file_names = [FLAGS.images_dir+f for f in os.listdir(FLAGS.images_dir) if re.search('jpg|JPG', f)]
target_vectors = {}
try:
    with open(flags.mapping_file, "r") as f:
        for line in f:
            # first column contains image name, remainder contains vector
            columns = line.split(',')
            vector = columns[1:1+FLAGS.n_dim]
            target_vectors[columns[0]] = np.array(vector)
except Exception:
    print("Could not find mapping file, using random labels instead.")
    for file_name in image_file_names:
        image_name = file_name.split('/')[-1].split('.')[0]
        target_vectors[image_name] = np.random.rand(FLAGS.n_dim)

result = {}

for file_name in image_file_names:
    image_name = file_name.split('/')[-1].split('.')[0]
    print("processing {0}".format(image_name))
    image_data = gfile.FastGFile(file_name, 'rb').read()
    decoder = tf.image.decode_jpeg(image_data, channels = 3)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        decoded_image = session.run(decoder)
    augmented_images = augment_image(decoded_image)
    result[image_name] = (augmented_images, target_vectors[image_name], image_data)
            
pickle.dump(result, open(os.path.join(FLAGS.output_dir, 'augmented'), 'wb'))