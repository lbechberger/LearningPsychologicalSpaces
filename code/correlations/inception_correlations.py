# -*- coding: utf-8 -*-
"""
Uses the feature vectors of the inception network to compute similarity ratings between images.
Checks how correlated they are to the original human similarity ratings.

Created on Sun May 12 07:56:40 2019

@author: lbechberger
"""

import pickle, argparse, os, sys, tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
from six.moves import urllib
from code.util import compute_correlations, distance_functions

parser = argparse.ArgumentParser(description='Pixel-based similarity baseline')
parser.add_argument('model_dir', help = 'folder for storing the pretrained network')
parser.add_argument('similarity_file', help = 'the input file containing the target similarity ratings')
parser.add_argument('image_folder', help = 'the folder containing the original images')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'create scatter plots of distances vs. dissimilarities')
args = parser.parse_args()

# set up file name for output file
_, path_and_file = os.path.splitdrive(args.similarity_file)
_, file = os.path.split(path_and_file)
file_without_extension = file.split('.')[0]
output_file_name = os.path.join(args.output_folder, "{0}.csv".format(file_without_extension))

# load the real similarity data
with open(args.similarity_file, 'rb') as f:
    input_data = pickle.load(f)

item_ids = input_data['items']
target_dissimilarities = input_data['dissimilarities']

# load all images
images = []
for item_id in item_ids:
    for file_name in os.listdir(args.image_folder):
        if os.path.isfile(os.path.join(args.image_folder, file_name)) and item_id in file_name:
            # found the corresponding image: load it
            image_data = gfile.FastGFile(os.path.join(args.image_folder, file_name), 'rb').read()            
            images.append(image_data)

inception_features = []

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
            features.append(feature_vector.reshape(1, -1))
            
    return features


maybe_download_and_extract()
create_graph()

inception_features = extract_inception_features(images)

print('extracted features')

with open(output_file_name, 'w', buffering=1) as f:

    f.write("scoring,pearson,spearman,kendall,r2_linear,r2_isotonic\n")
           
    for distance_name, distance_function in distance_functions.items():
        
        correlation_metrics = compute_correlations(inception_features, target_dissimilarities, distance_function)
        f.write("{0},{1},{2},{3},{4},{5}\n".format(distance_name, 
                                                    correlation_metrics['pearson'], 
                                                    correlation_metrics['spearman'], 
                                                    correlation_metrics['kendall'], 
                                                    correlation_metrics['r2_linear'], 
                                                    correlation_metrics['r2_isotonic']))

        print('done with {0}'.format(distance_name))
        
#        if args.plot:
#            # create scatter plot if user want us to
#            fig, ax = plt.subplots(figsize=(12,12))
#            ax.scatter(sim_vector,target_vector)
#            plt.xlabel('ANN-based Distance', fontsize = 20)
#            plt.ylabel('Dissimilarity from Psychological Study', fontsize = 20)
#                    
#            output_file_name = os.path.join(args.output_folder, '{0}.png'.format(scoring_name))        
#            
#            fig.savefig(output_file_name, bbox_inches='tight', dpi=200)
#            plt.close()