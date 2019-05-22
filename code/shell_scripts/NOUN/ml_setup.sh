#!/bin/bash

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/dataset/augmented data/NOUN/analysis/features 


# machine learning: preparation
echo 'preparing data set for machine learning'

# create more artificial images
echo '    augmentation'
python code/dataset/data_augmentation.py data/NOUN/images/ data/NOUN/dataset/augmented 1000 -s 42 --flip_prob 0.0 --crop_size 0.05 --scale_min 0.9 --scale_max 1.1 --translation 0.1 --sp_noise_prob 0.01 --rotation_angle 15

# collect regression targets
echo '    regression targets'
python code/dataset/prepare_targets.py data/NOUN/dataset/targets.csv data/NOUN/dataset/targets.pickle -s 42

# compute features
echo '    feature extraction'
echo '        inception network'
python code/regression/inception_features.py /tmp/inception data/NOUN/dataset/augmented data/NOUN/dataset/features_inception.pickle
echo '        reduced images'
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_mean_6_grey.pickle -b 6 -a mean -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_mean_30_grey.pickle -b 30 -a mean -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_mean_12_rgb.pickle -b 12 -a mean
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_mean_50_rgb.pickle -b 50 -a mean

# analyze feature spaces
echo '    cluster analysis'
echo '        inception network'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_inception.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_inception.txt
echo '        reduced images'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_mean_6_grey.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_mean_6_grey.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_mean_30_grey.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_mean_30_grey.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_mean_12_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_mean_12_rgb.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_mean_50_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_mean_50_rgb.txt

