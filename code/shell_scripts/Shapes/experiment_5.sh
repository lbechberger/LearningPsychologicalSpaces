#!/bin/bash

echo 'experiment 5 - generalizing to other target spaces'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_dims=("1 2 3 5 6 7 8 9 10")
default_image_size=224
default_epochs=200
default_patience=200
default_ann_config="-e -c 1.0 -r 0.0 -m 0.0625"
default_inception_lasso=0.005
default_transfer_features="small"
default_transfer_lasso=0.02

folds="${folds:-$default_folds}"
dims="${dims:-$default_dims}"
image_size="${image_size:-$default_image_size}"
epochs="${epochs:-$default_epochs}"
patience="${patience:-$default_patience}"
ann_config="${ann_config:-$default_ann_config}"
inception_lasso="${inception_lasso:-$default_inception_lasso}"
transfer_features="${transfer_features:-$default_transfer_features}"
transfer_lasso="${transfer_lasso:-$default_transfer_lasso}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	ann_script=code.ml.ann.run_ann
	regression_script=code.ml.regression.regression
	walltime=''
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	ann_script=code/ml/ann/run_ann.sge
	regression_script=code/ml/regression/regression.sge
	walltime='--walltime 5400'
	qsub ../Utilities/watch_jobs.sge $ann_script ann ../sge-logs/
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_5/logs/' 'data/Shapes/ml/experiment_5/snapshots/' 'data/Shapes/ml/experiment_5/aggregated' 'data/Shapes/ml/experiment_5/transfer' 'data/Shapes/ml/experiment_5/inception'


# start ann configurations (multi-task learning)
for dim in $dims
do
	for fold in $folds
	do
		$cmd $ann_script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_5/multi-task.csv $ann_config -s 42 -f $fold $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs
	done
done


# run lasso regression (inception)
for dim in $dims
do
	$cmd $regression_script data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" 'data/Shapes/ml/dataset/pickle/features_0.1.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_5/inception/mean_'"$dim"'.csv' -s 42 -e data/Shapes/ml/dataset/pickle/features_0.0.pickle --lasso $inception_lasso
done

# run lasso regression (transfer learning)
for dim in $dims
do
	for fold in $folds
	do
		$cmd $regression_script data/Shapes/ml/dataset/targets.pickle 'mean_'"$dim" 'data/Shapes/ml/experiment_3/features/'"$transfer_features"'_f'"$fold"'_noisy.pickle' data/Shapes/ml/dataset/pickle/folds.csv 'data/Shapes/ml/experiment_5/transfer/mean_'"$dim"'_f'"$fold"'.csv' -s 42 -e 'data/Shapes/ml/experiment_3/features/'"$transfer_features"'_f'"$fold"'_clean.pickle' --lasso $transfer_lasso
	done
done


# aggregate results for increased convenience
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_5/multi-task.csv data/Shapes/ml/experiment_5/aggregated/multi-task.csv
for dim in $dims
do
	python -m code.ml.regression.average_folds 'data/Shapes/ml/experiment_5/transfer/mean_'"$dim"'_f{0}.csv' 5 'data/Shapes/ml/experiment_5/aggregated/transfer_mean_'"$dim"'.csv'
done

