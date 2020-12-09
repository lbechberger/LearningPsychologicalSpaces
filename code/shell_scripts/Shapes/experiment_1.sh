#!/bin/bash

echo 'experiment 1'

# declare some lists to make code below less repetitive 
default_baselines=("--zero")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")

baselines="${baselines_ex1:-$default_baselines}"
regressors="${regressors_ex1:-$default_regressors}"
lassos="${lassos:-$default_lassos}"

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
mkdir -p data/Shapes/ml/experiment_1

# first compute the baselines
echo '    baselines'
for baseline in $baselines
do
	echo "        $baseline"	
	$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/ml/dataset/pickle/features_0.1.pickle data/Shapes/ml/dataset/pickle/folds.csv data/Shapes/ml/experiment_1/baselines.csv -s 42 $baseline
done

# now compute the results for a linear regression; also compute results on shuffled targets for comparison
echo '    regressors'
for regressor in $regressors
do
	echo "        $regressor"
	$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/ml/dataset/pickle/features_0.1.pickle data/Shapes/ml/dateset/pickle/folds.csv data/Shapes/ml/experiment_1/regression.csv -s 42 --shuffled $regressor
done


# finally do a grid search on the lasso regressor for the two selected feature sets (only correct targets)
echo '    lasso regressor'
for lasso in $lassos
do
	echo "        lasso $lasso"
	$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/ml/dataset/pickle/features_0.1.pickle data/Shapes/ml/dateset/pickle/folds.csv data/Shapes/ml/experiment_1/lasso.csv -s 42 --lasso $lasso
done
