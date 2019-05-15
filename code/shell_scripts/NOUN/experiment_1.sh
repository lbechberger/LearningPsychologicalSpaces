#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_1
mkdir -p data/NOUN/ML_results/experiment_1

echo 'experiment 1'
# first use the inception features; here also once compute baselines (they are independent of the feature space, so recomputing them every time does not make too much sense)
echo '    inception features'
echo '        zero'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --zero
echo '        mean'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --mean
echo '        normal'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --normal -r 10
echo '        draw'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --draw -r 10
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --linear
echo '        lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --alpha 0.25
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --alpha 0.5
echo '        lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --alpha 1.0
echo '        lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --alpha 2.0

# for features based on reduced images: only do linear regression and lasso regression
echo '    reduced images: min 7 grey'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --linear
echo '        lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --alpha 0.25
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --alpha 0.5
echo '        lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --alpha 1.0
echo '        lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --alpha 2.0

echo '    reduced images: min 18 grey'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --linear
echo '        lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --alpha 0.25
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --alpha 0.5
echo '        lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --alpha 1.0
echo '        lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --alpha 2.0


echo '    reduced images: min 12 rgb'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --linear
echo '        lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --alpha 0.25
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --alpha 0.5
echo '        lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --alpha 1.0
echo '        lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --alpha 2.0

echo '    reduced images: min 18 rgb'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --linear
echo '        lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --alpha 0.25
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --alpha 0.5
echo '        lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --alpha 1.0
echo '        lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --alpha 2.0
