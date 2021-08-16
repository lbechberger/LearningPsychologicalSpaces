#!/bin/bash

echo 'experiment 9 - generalizing autoencoder results to other target spaces'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_dims=("1 2 3 5 6 7 8 9 10")
default_image_size=224
default_epochs=200
default_patience=200
default_ann_config="-c 0.0 -r 1.0 -m 0.0625 -w 0.0"
default_transfer_features="best"
default_transfer_lasso=0.02

folds="${folds:-$default_folds}"
dims="${dims:-$default_dims}"
image_size="${image_size:-$default_image_size}"
epochs="${epochs:-$default_epochs}"
patience="${patience:-$default_patience}"
ann_config="${ann_config:-$default_ann_config}"
transfer_features="${transfer_features:-$default_transfer_features}"
transfer_lasso="${transfer_lasso:-$default_transfer_lasso}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	regression_script=code.ml.regression.regression
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	regression_script=code/ml/regression/regression.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# run lasso regression (transfer learning)
for dim in $dims
do
	for fold in $folds
	do
		$cmd $regression_script data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" 'data/Shapes/ml/experiment_7/features/'"$transfer_features"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_9/transfer/mean_'"$dim"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_7/features/'"$transfer_features"'_f'"$fold"'_clean.pickle' --lasso $transfer_lasso
	done
done

for dim in $dims
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_9/transfer/mean_'"$dim"'_f{0}.csv' 5 'data/Shapes/ml/experiment_9/aggregated/transfer_mean_'"$dim"'.csv'
done

