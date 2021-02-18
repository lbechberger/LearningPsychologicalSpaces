#!/bin/bash

echo 'experiment 3 - regression on top of sketch classification'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default accuracy correlation small")

folds="${folds:-$default_folds}"
regressors="${regressors_ex1:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features:-$default_features}"

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


# define snapshots of default classifier
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep25_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f0.pickle' > data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep21_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f1.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep55_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f2.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep28_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f3.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep31_FINAL.h5 data/Shapes/ml/experiment_3/features/default_f4.pickle' >> data/Shapes/ml/experiment_3/snapshots.config

# define snapshots of classifier with highest accuracy
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep51_FINAL.h5 data/Shapes/ml/experiment_3/features/accuracy_f0.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep71_FINAL.h5 data/Shapes/ml/experiment_3/features/accuracy_f1.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep47_FINAL.h5 data/Shapes/ml/experiment_3/features/accuracy_f2.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep92_FINAL.h5 data/Shapes/ml/experiment_3/features/accuracy_f3.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep51_FINAL.h5 data/Shapes/ml/experiment_3/features/accuracy_f4.pickle' >> data/Shapes/ml/experiment_3/snapshots.config

# define snapshots of classifier with highest correlation
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.25_mean_4_f0_ep90_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f0.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.25_mean_4_f1_ep85_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f1.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.25_mean_4_f2_ep86_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f2.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.25_mean_4_f3_ep95_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f3.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.001_v0.0_eTrue_dFalse_n0.25_mean_4_f4_ep83_FINAL.h5 data/Shapes/ml/experiment_3/features/correlation_f4.pickle' >> data/Shapes/ml/experiment_3/snapshots.config

# define snapshots of classifier with best results for small bottleneck
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b64_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep96_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f0.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b64_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep88_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f1.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b64_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep96_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f2.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b64_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep93_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f3.pickle' >> data/Shapes/ml/experiment_3/snapshots.config
echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b64_w0.001_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep95_FINAL.h5 data/Shapes/ml/experiment_3/features/small_f4.pickle' >> data/Shapes/ml/experiment_3/snapshots.config


# extract all the features
while IFS= read -r config
do
	$cmd $bottleneck_script data/Shapes/ml/dataset/Shapes.pickle $config -s 42
done < 'data/Shapes/ml/experiment_3/snapshots.config'


# run the regression
for feature in $features
do
	for fold in $folds
	do
		for regressor in $regressors
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/'"$feature"'_f'"$fold"'.csv' -s 42 $regressor
		done

		for lasso in $lassos
		do
			$cmd $regression_script data/Shapes/ml/dataset/targets.pickle mean_4 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_3/'"$feature"'_f'"$fold"'.csv' -s 42 --lasso $lasso
		done

	done
done

# do a cluster analysis
for feature in $features
do
	for fold in $folds
	do
		python -m code.ml.regression.cluster_analysis 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'.pickle' -n 100 -s 42 > 'data/Shapes/ml/experiment_3/features/'"$feature"'_f'"$fold"'.txt'
	done
done
python -m code.ml.regression.cluster_analysis 'data/Shapes/ml/dataset/pickle/features_0.1.pickle' -n 100 -s 42 > 'data/Shapes/ml/experiment_3/features/inception_0.1.txt'


# average the results across all folds for increased convenience
for feature in $features
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_3/'"$feature"'_f{0}.csv' 5 'data/Shapes/ml/experiment_3/aggregated/'"$feature"'.csv'
done
