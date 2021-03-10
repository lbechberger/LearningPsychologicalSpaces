#!/bin/bash

echo 'experiment 1 - inception baseline'

# declare some lists to make code below less repetitive 
default_baselines=("--zero")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_noises=("0.1 0.25 0.55")
default_best_noise=0.1
default_comparison_noises=("0.0 0.1")
default_dims=("1 2 3 5 6 7 8 9 10")

baselines="${baselines_ex1:-$default_baselines}"
regressors="${regressors_ex1:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
noises="${noises:-$default_noises}"
best_noise="${best_noise:-$default_best_noise}"
comparison_noises="${comparison_noises:-$default_comparison_noises}"
dims="${dims:-$default_dims}"

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
for noise in $default_noises
do
	mkdir -p 'data/Shapes/ml/experiment_1/noise_'"$noise"'/'
done
mkdir -p 'data/Shapes/ml/experiment_1/noise_0.0/'

# first analyze the mean and median 4d spaces
echo '    mean_4 and median_4 (noise and regression)'
for noise in $default_noises
do
	for baseline in $baselines
	do
		echo "        $baseline"	
		$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/mean_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
		$cmd $script data/Shapes/ml/dataset/targets.pickle median_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/median_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
	done
	
	for regressor in $regressors
	do
		echo "        $regressor"
		$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/mean_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --shuffled $regressor
		$cmd $script data/Shapes/ml/dataset/targets.pickle median_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/median_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --shuffled $regressor

	done

done

# compare performance to same train and test noise (either none or best noise setting)
echo '    performance comparison: same train and test noise'

for noise in $comparison_noises
do
	for regressor in $regressors
	do
		echo "        $regressor"
		$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/mean_4_same_noise.csv' -s 42 --shuffled $regressor
		$cmd $script data/Shapes/ml/dataset/targets.pickle median_4 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/median_4_same_noise.csv' -s 42 --shuffled $regressor
	done
done



# now run the regression for the other target spaces using the selected noise level (only correct targets)
for dim in $dims
do
	for baseline in $baselines
	do
		echo "        $baseline"	
		$cmd $script data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/mean_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
		$cmd $script data/Shapes/ml/dataset/targets.pickle 'median_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/median_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
	done
	
	for regressor in $regressors
	do
		echo "        $regressor"
		$cmd $script data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/mean_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $regressor
		$cmd $script data/Shapes/ml/dataset/targets.pickle 'median_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/median_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $regressor
	done

done


# finally do a grid search on the lasso regressor for the selected noise level (only correct targets)
echo '    lasso regressor on mean_4 and median_4'
for lasso in $lassos
do
	echo "        lasso $lasso"
	$cmd $script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/mean_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --lasso $lasso
	$cmd $script data/Shapes/ml/dataset/targets.pickle median_4 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/median_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --lasso $lasso
done
