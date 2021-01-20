#!/bin/bash

# set up global variables
default_export_size=256
default_image_size=112

export_size="${export_size:-$default_export_size}"
image_size="${image_size:-$default_image_size}"


# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/images/Berlin
mkdir -p data/Shapes/ml/dataset/0 data/Shapes/ml/dataset/1 data/Shapes/ml/dataset/2 data/Shapes/ml/dataset/3 data/Shapes/ml/dataset/4
mkdir -p data/Shapes/ml/dataset/pickle/0.1/ data/Shapes/ml/dataset/pickle/0.25/ data/Shapes/ml/dataset/pickle/0.55/
mkdir -p data/Shapes/ml/snapshots/

# exporting Berlin data manually
for folder in data/Shapes/images/Berlin-svg/*
do
	echo $folder
	output_folder=`echo $folder | sed -e 's/Berlin-svg/Berlin/'`
	mkdir -p $output_folder
	for file in $folder/*
	do
		output_path=`echo $file | sed -e 's/svg$/png/' | sed -e 's/Berlin-svg/Berlin/'`
		inkscape --export-png=$output_path --export-width=$export_size --export-height=$export_size --export-background=white $file > /dev/null
	done
done

# machine learning: preparation
echo 'preparing data set for machine learning'

# create more artificial images
echo '    augmentation'
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Shapes.csv data/Shapes/ml/dataset/ 2000 -s 42 -p data/Shapes/ml/dataset/pickle/ -n 0.1 0.25 0.55 > data/Shapes/ml/dataset/Shapes_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Additional.csv data/Shapes/ml/dataset/ 2000 -s 42 > data/Shapes/ml/dataset/Additional_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Berlin.csv data/Shapes/ml/dataset/ 12 -s 42 > data/Shapes/ml/dataset/Berlin_stat.txt
python -m code.ml.preprocessing.prepare_Shapes_data data/Shapes/ml/folds/Sketchy.csv data/Shapes/ml/dataset/ 4 -s 42 > data/Shapes/ml/dataset/Sketchy_stat.txt


# collect regression targets
echo '    regression targets'
python -m code.ml.preprocessing.prepare_targets data/Shapes/ml/regression_targets.csv data/Shapes/ml/dataset/targets.pickle -s 42

# compute features
echo '    feature extraction'
echo '        ANN-based features'
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.1/ data/Shapes/ml/dataset/pickle/features_0.1.pickle
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.25/ data/Shapes/ml/dataset/pickle/features_0.25.pickle
python -m code.ml.regression.ann_features /tmp/inception data/Shapes/ml/dataset/pickle/0.55/ data/Shapes/ml/dataset/pickle/features_0.55.pickle


