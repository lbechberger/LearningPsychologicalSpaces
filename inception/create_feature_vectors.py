# -*- coding: utf-8 -*-
"""
Script for mapping our images to the 2048-dimensional bottleneck layer of 
the inception-v3 network.

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

FLAGS = flags.FLAGS

def create_graph():
    with gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []
    
    create_graph()
    
    with tf.Session() as sess:
    
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for ind, image in enumerate(list_images):
            if (ind%10 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            # TODO: use real mapping here instead of random
            #image_name = image.split('/')[-1].split('.')[0]
            labels.append(np.random.rand(4))
            
    return features, labels

list_images = [FLAGS.images_dir+f for f in os.listdir(FLAGS.images_dir) if re.search('jpg|JPG', f)]
features,labels = extract_features(list_images)

pickle.dump(features, open(os.path.join(FLAGS.output_dir, 'features'), 'wb'))
pickle.dump(labels, open(os.path.join(FLAGS.output_dir, 'labels'), 'wb'))