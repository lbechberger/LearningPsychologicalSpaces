#!/bin/bash

echo 'experiment 8 - autoencoding and mapping'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_mapping_weights=("0.0625 0.125 0.25 0.5 1 2")
default_image_size=224
default_epochs=200
default_patience=200

folds="${folds:-$default_folds}"
mapping_weights="${mapping_weights:-$default_mapping_weights}"
image_size="${image_size:-$default_image_size}"
epochs="${epochs:-$default_epochs}"
patience="${patience:-$default_patience}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	script=code.ml.ann.run_ann
	walltime=''
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	script=code/ml/ann/run_ann.sge
	walltime='--walltime 5400'
	qsub ../Utilities/watch_jobs.sge $script ann ../sge-logs/
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_8/logs/' 'data/Shapes/ml/experiment_8/snapshots/' 'data/Shapes/ml/experiment_8/aggregated'


# define ann configurations to run
echo 'data/Shapes/ml/experiment_8/default.csv -e' > data/Shapes/ml/experiment_8/ann.config
echo 'data/Shapes/ml/experiment_8/best.csv -w 0.0' >> data/Shapes/ml/experiment_8/ann.config

# run all the configurations
while IFS= read -r config
do
	for mapping_weight in $mapping_weights
	do
		for fold in $folds
		do
			$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle $config -c 0.0 -r 1.0 -m $mapping_weight -s 42 -f $fold $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs
		done
	done
done < 'data/Shapes/ml/experiment_8/ann.config'
