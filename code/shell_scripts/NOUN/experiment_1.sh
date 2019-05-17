#!/bin/bash

echo 'experiment 1'

# declare some lists to make code below less repetitive 
feature_sets=("inception image_min_7_g image_min_18_g image_min_12_rgb image_min_18_rgb")
lasso_sets=("inception")
baselines=("--zero --mean --normal --draw")
regressors=("--linear --random_forest")
lassos=("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_1
mkdir -p data/NOUN/ML_results/experiment_1

# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
echo '    baselines'
for baseline in $baselines
do
	echo "        $baseline"	
	python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_1/inception.csv -s 42 $baseline
done

# now compute the results for a linear regression and a random forest regression on all feature sets; also compute results on shuffled targets for comparison
echo '    regressors'
for feature_set in $feature_sets
do
	echo "        $feature_set"
	for regressor in $regressors
	do
		echo "            $regressor"
		python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 'data/NOUN/dataset/features_'"$feature_set"'.pickle' 'data/NOUN/ML_results/experiment_1/'"$feature_set"'.csv' -s 42 --shuffled $regressor
	done
done

# finally do a grid search on the lasso regressor for the two selected feature sets
echo '    lasso regressor'
for feature_set in $lasso_sets
do
	echo "        $feature_set"
	for lasso in $lassos
	do
		echo "            lasso $lasso"
		python code/regression/regression.py data/NOUN/dataset/targets.pickle HorstHout_4 'data/NOUN/dataset/features_'"$feature_set"'.pickle' 'data/NOUN/ML_results/experiment_1/'"$feature_set"'.csv' -s 42 --lasso $lasso
	done
done

