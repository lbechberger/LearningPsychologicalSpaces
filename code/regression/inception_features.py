# -*- coding: utf-8 -*-
"""
Extracts features of the augmented images based on the inception-v3 network.

Based on https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

Created on Tue Jan 30 10:49:25 2018

@author: lbechberger
"""

import os, sys, tarfile, argparse, pickle
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from six.moves import urllib

parser = argparse.ArgumentParser(description='Feature Extraction with inception-v3')
parser.add_argument('model_dir', help = 'folder for storing the pretrained network')
parser.add_argument('input_dir', help = 'input folder containing the augmented data')
parser.add_argument('output_file', help = 'output file for the feature vectors')
args = parser.parse_args()

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract():
  """Download and extract model tar file."""
  
  if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(args.model_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(args.model_dir)


def create_graph():
    """Load the computation graph of the inception network."""
    with gfile.FastGFile(os.path.join(args.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_inception_features(images):
    """Extracts the inception features for the given images (assumed to be represented as one big tensor)."""
    
    features = []
    
    with tf.Session() as sess:
        second_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for image in images:
            feature_vector = sess.run(second_to_last_tensor, {'DecodeJpeg/contents:0': image})
            features.append(np.squeeze(feature_vector))
                    
    return features


maybe_download_and_extract()
print("Downloaded network")
create_graph()
print("imported graph")

result = {}
image_file_names = [f for f in os.listdir(args.input_dir)]

for image_file in image_file_names:
    image_name = image_file.split('.')[0]
    print("processing {0}".format(image_name))
    image_data = pickle.load(open(os.path.join(args.input_dir, image_file), 'rb'))
    result[image_name] = extract_inception_features(image_data)

pickle.dump(result, open(args.output_file, 'wb'))
