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
default_targets=("mean median")
default_shuffled_flag="--shuffled"

baselines="${baselines:-$default_baselines}"
regressors="${regressors:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
noises="${inception_noises:-$default_noises}"
best_noise="${best_noise:-$default_best_noise}"
comparison_noises="${comparison_noises:-$default_comparison_noises}"
dims="${dims:-$default_dims}"
targets="${targets:-$default_targets}"
shuffled_flag="${shuffled_flag:-$default_shuffled_flag}"

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

# first analyze the 4d space(s)
echo '    noise types for 4D space'
for noise in $default_noises
do
	for target in $targets
	do
		for baseline in $baselines
		do
			echo "        $baseline"	
			$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_4' 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/'"$target"'_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
		done
	
		for regressor in $regressors
		do
			echo "        $regressor"
			$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_4' 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/'"$target"'_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $shuffled_flag $regressor

		done
	done
done

# compare performance to same train and test noise (either none or best noise setting)
echo '    performance comparison: same train and test noise'

for noise in $comparison_noises
do
	for target in $targets
	do
		for regressor in $regressors
		do
			echo "        $regressor"
			$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_4' 'data/Shapes/ml/dataset/pickle/features_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$noise"'/'"$target"'_4_same_noise.csv' -s 42 $regressor
		done
	done

	python -m code.ml.regression.cluster_analysis 'data/Shapes/ml/dataset/pickle/features_'$noise'.pickle' -n 100 -s 42 > 'data/Shapes/ml/experiment_1/noise_'"$noise"'/cluster_analysis.txt'

done



# now run the regression for the other target spaces using the selected noise level (only correct targets)
echo '    other dimensions'
for dim in $dims
do
	for target in $targets
	do
		for baseline in $baselines
		do
			echo "        $baseline"	
			$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/'"$target"'_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $baseline
		done
	
		for regressor in $regressors
		do
			echo "        $regressor"
			$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_'"$dim" 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/'"$target"'_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle $regressor
		done
	done
done


# finally do a grid search on the lasso regressor for the selected noise level (only correct targets)
echo '    lasso regressor on 4D space(s)'
for target in $targets
do
	for lasso in $lassos
	do
		echo "        lasso $lasso"
		$cmd $script data/Shapes/ml/dataset/targets.pickle "$target"'_4' 'data/Shapes/ml/dataset/pickle/features_'"$best_noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_1/noise_'"$best_noise"'/'"$target"'_4.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --lasso $lasso
	done
done
