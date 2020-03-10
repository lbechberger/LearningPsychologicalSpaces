#!/bin/bash

echo 'experiment 2'

# declare some lists to make code below less repetitive 
default_feature_sets=("ANN pixel_1875")
default_lasso_sets=("ANN")
default_baselines=("--zero")
default_regressors=("--linear --random_forest")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_targets=("classical Kruskal metric_SMACOF nonmetric_SMACOF")

feature_sets="${feature_sets_ex2:-$default_feature_sets}"
lasso_sets="${lasso_sets_ex2:-$default_lasso_sets}"
baselines="${baselines_ex2:-$default_baselines}"
regressors="${regressors_ex2:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
targets="${targets_ex2:-$default_targets}"


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
mkdir -p data/NOUN/ml/experiment_2


for target in $targets
do
	echo "    $target"
	mkdir -p 'data/NOUN/ml/experiment_2/'"$target"'/'

	# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
	echo '        baselines'
	for baseline in $baselines
	do
		echo "            $baseline"	
		$cmd $script data/NOUN/ml/dataset/targets.pickle "$target"'_4' data/NOUN/ml/dataset/features_ANN.pickle data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_2/'"$target"'/baselines.csv' -s 42 $baseline
	done

	# now compute the results for a linear regression and a random forest regression on all feature sets; also compute results on shuffled targets for comparison
	echo '        regressors'
	for feature_set in $feature_sets
	do
		echo "            $feature_set"
		for regressor in $regressors
		do
			echo "                $regressor"
			$cmd $script data/NOUN/ml/dataset/targets.pickle "$target"'_4' 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --shuffled $regressor
		done
	done

	for feature_set in $lasso_sets
	do
		echo "            $feature_set"
		for lasso in $lassos
		do
			echo "            lasso $lasso"
			$cmd $script data/NOUN/ml/dataset/targets.pickle "$target"'_4' 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_2/'"$target"'/'"$feature_set"'.csv' -s 42 --lasso $lasso
		done

	done
done

