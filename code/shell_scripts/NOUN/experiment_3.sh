#!/bin/bash

echo 'experiment 3'

# declare some lists to make code below less repetitive 
feature_sets=("inception image_min_7_g")
baselines=("--zero")
regressors=("--linear --random_forest")
lassos=("0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0"))
targets=("1 2 3 5 6 7 8 9 10")

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd=python
	script=code/regression/regression.py
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	script=code/regression/regression.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_3
mkdir -p data/NOUN/ML_results/experiment_3

for target in $targets
do
	echo '    Kruskal_$target'
	mkdir -p 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/'

	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		$cmd $script data/NOUN/dataset/targets.pickle 'Kruskal_'"$target" data/NOUN/dataset/features_inception.pickle data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for the real regressions
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			$cmd $script data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/'"$feature_set"'.csv' -s 42 $regressor
		done

		for lasso in $lassos
		do
			echo "            lasso $lasso"
			$cmd $script data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_3/Kruskal_'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done
