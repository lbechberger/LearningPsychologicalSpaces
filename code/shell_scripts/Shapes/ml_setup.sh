#!/bin/bash

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/ml/dataset/0 data/Shapes/ml/dataset/1 data/Shapes/ml/dataset/2 data/Shapes/ml/dataset/3 data/Shapes/ml/dataset/4


# machine learning: preparation
echo 'preparing data set for machine learning'

# create more artificial images
echo '    augmentation'
#python -m code.ml.preprocessing.data_augmentation data/NOUN/images/ data/NOUN/ml/dataset/augmented 1000 -s 42 --flip_prob 0.0 --crop_size 0.05 --scale_min 0.9 --scale_max 1.1 --translation 0.1 --sp_noise_prob 0.0 --rotation_angle 15

# collect regression targets
echo '    regression targets'
#python -m code.ml.preprocessing.prepare_targets data/NOUN/ml/targets.csv data/NOUN/ml/dataset/targets.pickle -s 42

# compute features
echo '    feature extraction'
echo '        ANN-based features'
#python -m code.ml.regression.ann_features /tmp/inception data/NOUN/ml/dataset/augmented data/NOUN/ml/dataset/features_ANN.pickle


