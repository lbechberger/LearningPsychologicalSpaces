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
import pickle
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(42) # make sure that we create the same set of augmentations every time

flags = tf.flags
flags.DEFINE_string('images_dir', '../images/', 'Location of data.')
flags.DEFINE_string('output_dir', 'features/augmented/', 'Where to store the feature vectors.')
flags.DEFINE_integer('n_samples', 1000, 'Number of augmented samples per original image.')

FLAGS = flags.FLAGS

# define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
        mode="constant", #fill with constant white pixels
        cval=255
    ),
    # add some salt and pepper noise (setting 3% of all pixels to 0 or 255)
    iaa.SaltAndPepper(0.03)
    ], random_order=True) # apply augmenters in random order

# define tensorflow graph: image --> string
tf_image = tf.placeholder(tf.uint8, shape=(300,300,3))
encoder = tf.image.encode_jpeg(tf_image)
# string --> image
tf_image_string = tf.placeholder(tf.string)
decoder = tf.image.decode_jpeg(tf_image_string)
    
def augment_image(base_image, num_samples):

    augmented_images = [] 
    for i in range(num_samples):
        image_aug = seq.augment_image(base_image)
        augmented_images.append(image_aug)

    result = []    
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for img in augmented_images:
            encoded_image = session.run(encoder, feed_dict = {tf_image : img})
            result.append(encoded_image)
    return result

image_file_names = [FLAGS.images_dir+f for f in os.listdir(FLAGS.images_dir) if re.search('jpg|JPG', f)]

for file_name in image_file_names:
    image_name = file_name.split('/')[-1].split('.')[0]
    print("processing {0}".format(image_name))
    image_data = gfile.FastGFile(file_name, 'rb').read()
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        decoded_image = session.run(decoder, feed_dict = {tf_image_string : image_data})
    augmented_images = augment_image(decoded_image, FLAGS.n_samples)
    pickle.dump(augmented_images, open(os.path.join(FLAGS.output_dir, image_name), 'wb'))