#!/bin/bash

echo 'experiment 3'

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_3
mkdir -p data/NOUN/ML_results/experiment_3

# declare some lists to make code below less repetitive 
feature_sets=("inception") #TODO: add best image-based one
baselines=("--zero --mean --normal --draw")
regressors=("--linear --random_forest")
lassos=("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
targets=("1 2 3 5 6 7 8 9 10")

for target in $targets
do
	echo '    Kruskal_$target'
	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		python code/regression/regression.py data/NOUN/dataset/targets.pickle 'Kruskal_'"$target" data/NOUN/dataset/features_inception.pickle 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for the real regressions
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			python code/regression/regression.py data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/'"$feature_set"'.csv' -s 42 $regressor
		done

		for lasso in $lassos
		do
			echo "            lasso $lasso"
			python code/regression/regression.py data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done


# TODO: for each dimensionality in best MDS strategy: run best baseline, best pixel, inception
