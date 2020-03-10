#!/bin/bash

echo 'experiment 3'

# declare some lists to make code below less repetitive 
default_feature_sets=("ANN")
default_lasso_sets=("ANN")
default_baselines=("--zero")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_targets=("1 2 3 5 6 7 8 9 10")

feature_sets="${feature_sets_ex3:-$default_feature_sets}"
lasso_sets="${lasso_sets_ex3:-$default_lasso_sets}"
baselines="${baselines_ex3:-$default_baselines}"
regressors="${regressors_ex3:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
targets="${targets_ex3:-$default_targets}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	script=code.ml.regression.regression
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	script=code/ml/regression/regression.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p data/NOUN/ml/experiment_3

for target in $targets
do
	echo "    nonmetric_SMACOF_$target"
	mkdir -p 'data/NOUN/ml/experiment_3/nonmetric_SMACOF_'"$target"'/'

	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		$cmd $script data/NOUN/ml/dataset/targets.pickle 'nonmetric_SMACOF_'"$target" data/NOUN/ml/dataset/features_ANN.pickle data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_3/nonmetric_SMACOF_'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for the real regressions
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			$cmd $script data/NOUN/ml/dataset/targets.pickle 'nonmetric_SMACOF_'"$target" 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_3/nonmetric_SMACOF_'"$target"'/'"$feature_set"'.csv' -s 42 $regressor
		done
	done

	for feature_set in $lasso_sets
	do
		echo "            $feature_set"
		for lasso in $lassos
		do
			echo "            lasso $lasso"
			$cmd $script data/NOUN/ml/dataset/targets.pickle 'nonmetric_SMACOF_'"$target" 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_3/nonmetric_SMACOF_'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done
