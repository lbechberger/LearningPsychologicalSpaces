#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

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
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_7_g.pickle -b 7 -a min -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_18_g.pickle -b 18 -a min -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_12_rgb.pickle -b 12 -a min
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_18_rgb.pickle -b 18 -a min

# analyze feature spaces
echo '    cluster analysis'
echo '        inception network'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_inception.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_inception.txt
echo '        reduced images'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_7_g.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_7_g.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_18_g.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_18_g.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_12_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_12_rgb.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_18_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_18_rgb.txt

