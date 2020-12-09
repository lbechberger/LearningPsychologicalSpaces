#!/bin/bash

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/ml/dataset/0 data/Shapes/ml/dataset/1 data/Shapes/ml/dataset/2 data/Shapes/ml/dataset/3 data/Shapes/ml/dataset/4
mkdir -p data/Shapes/ml/dataset/pickle/0.1/ data/Shapes/ml/dataset/pickle/0.25/ data/Shapes/ml/dataset/pickle/0.55/


# machine learning: preparation
echo 'preparing data set for machine learning'

# create more artificial images
echo '    augmentation'
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Shapes.csv data/Shapes/ml/dataset/ 4168 -s 42 -p data/Shapes/ml/dataset/pickle/ -n 0.1 0.25 0.55 > data/Shapes/ml/dataset/Shapes_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Additional.csv data/Shapes/ml/dataset/ 3520 -s 42 > data/Shapes/ml/dataset/Additional_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Berlin.csv data/Shapes/ml/dataset/ 25 -s 42 > data/Shapes/ml/dataset/Berlin_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Sketchy.csv data/Shapes/ml/dataset/ 8 -s 42 > data/Shapes/ml/dataset/Sketchy_stat.txt


# collect regression targets
echo '    regression targets'
python -m code.ml.preprocessing.prepare_targets data/Shapes/ml/regression_targets.csv data/Shapes/ml/dataset/targets.pickle -s 42

# compute features
echo '    feature extraction'
echo '        ANN-based features'
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.1/ data/Shapes/ml/dataset/pickle/features_0.1.pickle
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.25/ data/Shapes/ml/dataset/pickle/features_0.25.pickle
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.55/ data/Shapes/ml/dataset/pickle/features_0.55.pickle


