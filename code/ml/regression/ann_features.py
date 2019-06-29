# -*- coding: utf-8 -*-
"""
Extracts features of the augmented images based on the inception-v3 network.

Based on https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

Created on Tue Jan 30 10:49:25 2018

@author: lbechberger
"""

import os, argparse, pickle
from code.util import extract_inception_features

parser = argparse.ArgumentParser(description='Feature Extraction with inception-v3')
parser.add_argument('model_dir', help = 'folder for storing the pretrained network')
parser.add_argument('input_dir', help = 'input folder containing the augmented data')
parser.add_argument('output_file', help = 'output file for the feature vectors')
args = parser.parse_args()

result = {}
image_file_names = [f for f in os.listdir(args.input_dir)]

for image_file in image_file_names:
    image_name = image_file.split('.')[0]
    print("        processing {0}".format(image_name))
    image_data = pickle.load(open(os.path.join(args.input_dir, image_file), 'rb'))
    result[image_name] = extract_inception_features(image_data, args.model_dir, (-1))

pickle.dump(result, open(args.output_file, 'wb'))
