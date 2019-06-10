#!/bin/bash

echo 'experiment 2'

# declare some lists to make code below less repetitive 
feature_sets=("ANN")
lasso_sets=("ANN")
baselines=("--zero")
regressors=("--linear --random_forest")
lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
targets=("classical Kruskal metric_SMACOF nonmetric_SMACOF")

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
rm -r -f data/NOUN/ML_results/experiment_2
mkdir -p data/NOUN/ML_results/experiment_2


for target in $targets
do
	echo '    $target'
	mkdir -p 'data/NOUN/ML_results/experiment_2/'"$target"'/'

	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		$cmd $script data/NOUN/dataset/targets.pickle "$target"'_4' data/NOUN/dataset/features_ANN.pickle data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for a linear regression and a random forest regression on all feature sets; also compute results on shuffled targets for comparison
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			$cmd $script data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --shuffled $regressor
		done
	done

	for feature_set in $lasso_sets
	do
		echo "            $feature_set"
		for lasso in $lassos
		do
			echo "            lasso $lasso"
			$cmd $script data/NOUN/dataset/targets.pickle "$target"'_4' 'data/NOUN/dataset/features_'"$feature_set"'.pickle' data/NOUN/dataset/folds.csv 'data/NOUN/ML_results/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done

