#!/bin/bash

echo 'experiment 7 - regression on top of autoencoder'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default reconstruction correlation")
default_noises=("noisy clean")
default_image_size=224

folds="${folds:-$default_folds}"
regressors="${regressors:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features_exp7:-$default_features}"
noises="${noises_exp7:-$default_noises}"
image_size="${image_size:-$default_image_size}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	bottleneck_script=code.ml.ann.get_bottleneck_activations
	regression_script=code.ml.regression.regression
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	bottleneck_script=code/ml/ann/get_bottleneck_activations.sge
	regression_script=code/ml/regression/regression.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_7/features' 'data/Shapes/ml/experiment_7/aggregated'

# extract features for both noised and unnoised input
for noise in $noises
do
	if [ $noise = noisy ]
	then
		noise_flag="-n 0.1"
	else
		noise_flag=""
	fi

	# define snapshots of default autoencoder
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep95_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep25_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep48_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep56_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep52_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config


	# TODO define snapshots of classifier with highest correlation
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f0_ep4_FINAL.h5 data/Shapes/ml/experiment_7/features/correlation_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f1_ep5_FINAL.h5 data/Shapes/ml/experiment_7/features/correlation_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f2_ep6_FINAL.h5 data/Shapes/ml/experiment_7/features/correlation_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f3_ep4_FINAL.h5 data/Shapes/ml/experiment_7/features/correlation_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f4_ep4_FINAL.h5 data/Shapes/ml/experiment_7/features/correlation_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config


	# TODO define snapshots of autoencoder with best reconstruction
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep166_FINAL.h5 data/Shapes/ml/experiment_7/features/reconstruction_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep167_FINAL.h5 data/Shapes/ml/experiment_7/features/reconstruction_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep182_FINAL.h5 data/Shapes/ml/experiment_7/features/reconstruction_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep192_FINAL.h5 data/Shapes/ml/experiment_7/features/reconstruction_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep180_FINAL.h5 data/Shapes/ml/experiment_7/features/reconstruction_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config


done

# extract all the features
while IFS= read -r config
do
	$cmd $bottleneck_script data/Shapes/ml/dataset/Shapes.pickle $config -s 42 -i $image_size
done < 'data/Shapes/ml/experiment_7/snapshots.config'


# run the regression
for feature in $features
do
	for fold in $folds
	do
		for regressor in $regressors
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_7/'"$feature"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_clean.pickle' $regressor
		done

		for lasso in $lassos
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_7/'"$feature"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_clean.pickle' --lasso $lasso
		done

	done
done



# do a cluster analysis
for feature in $features
do
	for fold in $folds
	do
		for noise in $noises
		do
			python -m code.ml.regression.cluster_analysis 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_'"$noise"'.pickle' -n 100 -s 42 > 'data/Shapes/ml/experiment_7/features/'"$feature"'_f'"$fold"'_'"$noise"'.txt'
		done
	done
done



# average the results across all folds for increased convenience
for feature in $features
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_7/'"$feature"'_f{0}.csv' 5 'data/Shapes/ml/experiment_7/aggregated/'"$feature"'.csv'
done
