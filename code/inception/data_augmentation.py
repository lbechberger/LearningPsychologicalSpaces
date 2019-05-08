# -*- coding: utf-8 -*-
"""
Script for creating an augmented data set.

Inspired by https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence

Created on Tue Jan 30 10:49:25 2018

@author: lbechberger
"""

import os, re,  pickle, argparse
import imgaug as ia
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser(description='Data Augmentation')
parser.add_argument('images_dir', help = 'folder containing the original images')
parser.add_argument('output_dir', help = 'output folder for augmented data')
parser.add_argument('n', type = int, help = 'number of augmented samples per original image')
parser.add_argument('--flip_prob', type = float, help = 'probability of horizontal flips', default = 0.5)
parser.add_argument('--crop_size', type = float, help = 'maximal percentage of cropping for each image side', default = 0.1)
parser.add_argument('--blur_prob', type = float, help = 'probability of random blur', default = 0.5)
parser.add_argument('--blur_sigma', type = float, help = 'sigma of random blur', default = 0.5)
parser.add_argument('--contrast_min', type = float, help = 'minimal relative contrast', default = 0.75)
parser.add_argument('--contrast_max', type = float, help = 'maximal relative contrast', default = 1.5)
parser.add_argument('--g_noise_sigma', type = float, help = 'sigma of additive Gaussian noise', default = 0.05)
parser.add_argument('--g_noise_channel_prob', type = float, help = 'probability of different additive Gaussian noise for different color channels', default = 0.5)
parser.add_argument('--light_min', type = float, help = 'minimal relative brightness', default = 0.8)
parser.add_argument('--light_max', type = float, help = 'maximal relative brightness', default = 1.2)
parser.add_argument('--light_channel_prob', type = float, help = 'probability of different brightness change on different color channels', default = 0.2)
parser.add_argument('--scale_min', type = float, help = 'minimal relative size', default = 0.8)
parser.add_argument('--scale_max', type = float, help = 'maximal relative size', default = 1.2)
parser.add_argument('--translation', type = float, help = 'relative size of translation', default = 0.2)
parser.add_argument('--rotation_angle', type = int, help = 'maximal rotation angle', default = 25)
parser.add_argument('--shear_angle', type = int, help = 'maximal shear angle', default = 8)
parser.add_argument('--sp_noise_prob', type = float, help = 'probability of salt and pepper noise', default = 0.03)
parser.add_argument('-s', '--seed', type = int, help = 'seed for random number generation', default = None)
parser.add_argument('-i', '--image_size', type = int, help = 'expected size of input images', default = 300)
args = parser.parse_args()

if args.seed is not None:
    ia.seed(args.seed)

# define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(args.flip_prob), # horizontal flips
    iaa.Crop(percent=(0, args.crop_size)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(args.blur_prob,
        iaa.GaussianBlur(sigma=(0, args.blur_sigma))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((args.contrast_min, args.contrast_max)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, args.g_noise_sigma*255), per_channel=args.g_noise_channel_prob),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((args.light_min, args.light_max), per_channel=args.light_channel_prob),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (args.scale_min, args.scale_max), "y": (args.scale_min, args.scale_max)},
        translate_percent={"x": (-args.translation, args.translation), "y": (-args.translation, args.translation)},
        rotate=(-args.rotation_angle, args.rotation_angle),
        shear=(-args.shear_angle, args.shear_angle),
        mode="constant", #fill with constant white pixels
        cval=255
    ),
    # add some salt and pepper noise (setting 3% of all pixels to 0 or 255)
    iaa.SaltAndPepper(args.sp_noise_prob)
    ], random_order=True) # apply augmenters in random order

# define tensorflow graph: image --> string
tf_image = tf.placeholder(tf.uint8, shape=(args.image_size, args.image_size, 3))
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

image_file_names = [args.images_dir + f for f in os.listdir(args.images_dir) if re.search('jpg|JPG', f)]

for file_name in image_file_names:
    image_name = file_name.split('/')[-1].split('.')[0]
    print("processing {0}".format(image_name))
    image_data = gfile.FastGFile(file_name, 'rb').read()
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        decoded_image = session.run(decoder, feed_dict = {tf_image_string : image_data})
    augmented_images = augment_image(decoded_image, args.n)
    pickle.dump(augmented_images, open(os.path.join(args.output_dir, image_name), 'wb'))