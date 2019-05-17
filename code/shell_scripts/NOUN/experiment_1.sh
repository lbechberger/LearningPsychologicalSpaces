#!/bin/bash

echo 'experiment 1'

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_1
mkdir -p data/NOUN/ML_results/experiment_1

baselines=("--zero" "--mean" "--normal" "--draw")
max=5

for baseline in $baselines
do
	echo $baseline
	#python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/mean/Kruskal/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/mean/Kruskal/convexity/convexities.csv -b -r 100 -s 42 > 'data/Shapes/analysis/mean/Kruskal/convexity/'"$i"'D-convexity.txt' &
done


# first use the inception features; here also once compute baselines (they are independent of the feature space, so recomputing them every time does not make too much sense)
echo '    inception features'
echo '        zero'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --zero
echo '        mean'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --mean
echo '        normal'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --normal
echo '        draw'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --draw
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --linear
echo '        lasso 0.01'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.01
echo '        lasso 0.02'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.02
echo '        lasso 0.05'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.05
echo '        lasso 0.1'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.1
echo '        lasso 0.2'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.2
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 --lasso 0.5

# for features based on reduced images: only do linear regression and lasso regression
echo '    reduced images: min 7 grey'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --linear
echo '        lasso 0.01'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.01
echo '        lasso 0.02'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.02
echo '        lasso 0.05'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.05
echo '        lasso 0.1'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.1
echo '        lasso 0.2'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.2
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_7_g.pickle data/NOUN/ML_results/experiment_1/min_7_grey.csv -s 42 --lasso 0.5

echo '    reduced images: min 18 grey'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --linear
echo '        lasso 0.01'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.01
echo '        lasso 0.02'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.02
echo '        lasso 0.05'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.05
echo '        lasso 0.1'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.1
echo '        lasso 0.2'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.2
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_g.pickle data/NOUN/ML_results/experiment_1/min_18_grey.csv -s 42 --lasso 0.5


echo '    reduced images: min 12 rgb'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --linear
echo '        lasso 0.01'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.01
echo '        lasso 0.02'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.02
echo '        lasso 0.05'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.05
echo '        lasso 0.1'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.1
echo '        lasso 0.2'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.2
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_12_rgb.pickle data/NOUN/ML_results/experiment_1/min_12_rgb.csv -s 42 --lasso 0.5

echo '    reduced images: min 18 rgb'
echo '        linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --linear
echo '        lasso 0.01'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.01
echo '        lasso 0.02'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.02
echo '        lasso 0.05'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.05
echo '        lasso 0.1'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.1
echo '        lasso 0.2'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.2
echo '        lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_image_min_18_rgb.pickle data/NOUN/ML_results/experiment_1/min_18_rgb.csv -s 42 --lasso 0.5

