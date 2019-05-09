# -*- coding: utf-8 -*-
"""
Displays some of the augmented images. Can be used to visually check that augmentation is working fine.

Created on Wed May  8 12:12:26 2019

@author: lbechberger
"""

import argparse, pickle
import matplotlib.pyplot as plt
import tensorflow as tf

parser = argparse.ArgumentParser(description='Visualizing augmented images')
parser.add_argument('input_file', help = 'picke file containing the augmented images to display')
parser.add_argument('-r', '--rows', type = int, help = 'number of rows', default = 3)
parser.add_argument('-c', '--columns', type = int, help = 'number of columns', default = 4)
args = parser.parse_args()

with open(args.input_file, "rb") as f:
    images = pickle.load(f)

# need to convert tensorflow string representation into numbers
tf_image_string = tf.placeholder(tf.string)
decoder = tf.image.decode_jpeg(tf_image_string)

fig = plt.figure(figsize=(16,10))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for i in range(args.rows * args.columns):
        ax = fig.add_subplot(args.rows, args.columns, i+1)
        img = session.run(decoder, feed_dict = {tf_image_string : images[i]})
        ax.imshow(img)
    
plt.show()