#!/bin/bash

echo 'experiment 1'

# declare some lists to make code below less repetitive 
feature_sets=("ANN pixel_1875 pixel_507")
lasso_sets=("ANN pixel_1875 pixel_507")
baselines=("--zero --mean --normal --draw")
regressors=("--linear --random_forest")
lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")

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
mkdir -p data/NOUN/ml/experiment_1

# first compute the baselines: using a single feature set is sufficient as features are ignored anyways, don't need shuffled targets as results are same anyways
echo '    baselines'
for baseline in $baselines
do
	echo "        $baseline"	
	$cmd $script data/NOUN/ml/dataset/targets.pickle HorstHout_4 data/NOUN/ml/dataset/features_ANN.pickle data/NOUN/ml/folds.csv data/NOUN/ml/experiment_1/baselines.csv -s 42 $baseline
done

# now compute the results for a linear regression and a random forest regression on all feature sets; also compute results on shuffled targets for comparison
echo '    regressors'
for feature_set in $feature_sets
do
	echo "        $feature_set"
	for regressor in $regressors
	do
		echo "            $regressor"
		$cmd $script data/NOUN/ml/dataset/targets.pickle HorstHout_4 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_1/'"$feature_set"'.csv' -s 42 --shuffled $regressor
	done
done

# finally do a grid search on the lasso regressor for the two selected feature sets (only correct targets)
echo '    lasso regressor'
for feature_set in $lasso_sets
do
	echo "        $feature_set"
	for lasso in $lassos
	do
		echo "            lasso $lasso"
		$cmd $script data/NOUN/ml/dataset/targets.pickle HorstHout_4 'data/NOUN/ml/dataset/features_'"$feature_set"'.pickle' data/NOUN/ml/folds.csv 'data/NOUN/ml/experiment_1/'"$feature_set"'.csv' -s 42 --lasso $lasso
	done
done
