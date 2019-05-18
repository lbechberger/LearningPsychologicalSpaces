#!/bin/bash

echo 'experiment 2'

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_2
mkdir -p data/NOUN/ML_results/experiment_2/classical data/NOUN/ML_results/experiment_2/Kruskal 
mkdir -p data/NOUN/ML_results/experiment_2/metric_SMACOF data/NOUN/ML_results/experiment_2/nonmetric_SMACOF

# declare some lists to make code below less repetitive 
feature_sets=("inception") #TODO: add best image-based one
baselines=("--zero --mean --normal --draw")
regressors=("--linear --random_forest")
lassos=("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
targets=("classical Kruskal metric_SMACOF nonmetric_SMACOF")

for target in $targets
do
	echo '    $target'
	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		python code/regression/regression.py data/NOUN/dataset/targets.pickle "$target"'_4' data/NOUN/dataset/features_inception.pickle data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for a linear regression and a random forest regression on all feature sets; also compute results on shuffled targets for comparison
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			python code/regression/regression.py data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --shuffled $regressor
		done

		for lasso in $lassos
		do
			echo "            lasso $lasso"
			python code/regression/regression.py data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done

