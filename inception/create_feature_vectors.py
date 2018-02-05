# -*- coding: utf-8 -*-
"""
Script for mapping our images to the 2048-dimensional bottleneck layer of 
the inception-v3 network.

Inspired by https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

Created on Tue Jan 30 10:49:25 2018

@author: lbechberger
"""

import os
import sys
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle
from six.moves import urllib

flags = tf.flags
flags.DEFINE_string('model_dir', '/tmp/imagenet/', 'Directory where the pretrained network resides.')
flags.DEFINE_string('input_dir', 'features/augmented', 'Location of data.')
flags.DEFINE_string('output_dir', 'features/features', 'Where to store the feature vectors.')

FLAGS = flags.FLAGS

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_graph():
    """Load the computation graph of the inception network."""
    with gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_inception_features(images):
    """Extracts the inception features for the given images (assumed to be represented as one big tensor)."""
    features = []
    
    with tf.Session() as sess:
    
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for idx, image in enumerate(images):
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image})
            features.append(np.squeeze(predictions))
                    
    return features


maybe_download_and_extract()
print("Downloaded network")
create_graph()
print("imported graph")

input_data = {}
image_file_names = [f for f in os.listdir(FLAGS.input_dir)]
try:
    for image_name in image_file_names:
        image_data = pickle.load(open(os.path.join(FLAGS.input_dir, image_name), 'rb'))
        input_data[image_name] = image_data
except Exception:
    print("Cannot read augmented images. Aborting.")
    sys.exit(0)
print("fetched input data")

result = {}
    
for image_name, (augmented_images, target_vector, original_image) in input_data.items():
    print("processing {0}".format(image_name))
    augmented_features = extract_inception_features(augmented_images)
    original_features = extract_inception_features([original_image])
    result[image_name] = (augmented_features, target_vector, original_features)

pickle.dump(result, open(os.path.join(FLAGS.output_dir, 'features'), 'wb')) 