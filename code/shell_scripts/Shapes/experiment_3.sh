#!/bin/bash

echo 'experiment 3 - regression on top of sketch classification'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default large small correlation no_noise")
default_noises=("noisy clean")
default_image_size=224

folds="${folds:-$default_folds}"
regressors="${regressors:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features:-$default_features}"
noises="${noises_exp3:-$default_noises}"
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
mkdir -p 'data/Shapes/ml/experiment_3/features' 'data/Shapes/ml/experiment_3/aggregated'

# extract features for both noised and unnoised input
for noise in $noises
do
	if [ $noise = noisy ]
	then
		noise_flag="-n 0.1"
	else
		noise_flag=""
	fi

	# define snapshots of default classifier (has best accuracy)
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep196_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep188_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep179_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep198_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep177_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config


	# define snapshots of classifier with large bottleneck (2048 units)
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep194_FINAL.h5 data/Shapes/ml/experiment_3/features/large_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep142_FINAL.h5 data/Shapes/ml/experiment_3/features/large_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep182_FINAL.h5 data/Shapes/ml/experiment_3/features/large_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep144_FINAL.h5 data/Shapes/ml/experiment_3/features/large_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep183_FINAL.h5 data/Shapes/ml/experiment_3/features/large_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config


	# define snapshots of classifier with small bottleneck (256 units)
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep166_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep167_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep182_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep192_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep180_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config


	# define snapshots of classifier with highest correlation
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f0_ep4_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f1_ep5_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f2_ep6_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f3_ep4_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eFalse_dFalse_n0.1_mean_4_f4_ep4_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config

	# define snapshots of classifier trained w/o noise
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f0_ep165_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f1_ep184_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f2_ep169_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f3_ep165_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f4_ep132_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.config


done

# extract all the features
while IFS= read -r config
do
	$cmd $bottleneck_script data/Shapes/ml/dataset/Shapes.pickle $config -s 42 -i $image_size
done < 'data/Shapes/ml/experiment_3/snapshots.config'


# run the regression
for feature in $features
do
	for fold in $folds
	do
		for regressor in $regressors
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/'"$feature"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_clean.pickle' $regressor
		done

		for lasso in $lassos
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/'"$feature"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_clean.pickle' --lasso $lasso
		done

	done
done

# compare performance to same train and test noise (either none or best noise setting) for default and no_noise
echo '    performance comparison: same train and test noise'

for noise in $noises
do
	for fold in $folds
	do
		for regressor in $regressors
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/default_f'"$fold"'_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/default_f'"$fold"'_'"$noise"'.csv' -s 42 $regressor

			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/no_noise_f'"$fold"'_'"$noise"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/no_noise_f'"$fold"'_'"$noise"'.csv' -s 42 $regressor

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
			python -m code.ml.regression.cluster_analysis 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_'"$noise"'.pickle' -n 100 -s 42 > 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'_'"$noise"'.txt'
		done
	done
done



# average the results across all folds for increased convenience
for feature in $features
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_3/'"$feature"'_f{0}.csv' 5 'data/Shapes/ml/experiment_3/aggregated/'"$feature"'.csv'
done

for noise in $noises
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_3/default_f{0}_'"$noise"'.csv' 5 'data/Shapes/ml/experiment_3/aggregated/default_'"$noise"'.csv'
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_3/no_noise_f{0}_'"$noise"'.csv' 5 'data/Shapes/ml/experiment_3/aggregated/no_noise_'"$noise"'.csv'
done
