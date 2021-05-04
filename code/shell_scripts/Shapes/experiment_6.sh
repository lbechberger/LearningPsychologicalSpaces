#!/bin/bash

echo 'experiment 6 - reconstruction baseline'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_weight_decays_enc=("0.0 0.0002 0.001 0.002")
default_weight_decays_dec=("0.0002 0.0005 0.001 0.002")
default_noises=("0.0 0.25 0.55")
default_bottlenecks=("2048 256 128 64 32 16")
default_image_size=224
default_epochs=200
default_patience=200

folds="${folds:-$default_folds}"
weight_decays_enc="${weight_decays:-$default_weight_decays_enc}"
weight_decays_dec="${weight_decays_dec:-$default_weight_decays_dec}"
noises="${noises:-$default_noises}"
bottlenecks="${bottlenecks:-$default_bottlenecks}"
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
mkdir -p 'data/Shapes/ml/experiment_6/logs/' 'data/Shapes/ml/experiment_6/snapshots' 'data/Shapes/ml/experiment_6/aggregated'

# vanilla setup
for fold in $folds
do

	$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/default.csv -c 0.0 -r 1.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs 
done

# weight decay encoder
for weight_decay in $weight_decays_enc
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/decay_enc.csv -c 0.0 -r 1.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs -w $weight_decay
	done
done

# weight decay decoder
for weight_decay in $weight_decays_dec
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/decay_dec.csv -c 0.0 -r 1.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs -v $weight_decay
	done
done

# no dropout encoder
for fold in $folds
do
	$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/dropout_enc.csv -c 0.0 -r 1.0 -m 0.0 -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs
done

# dropout decoder
for fold in $folds
do
	$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/dropout_dec.csv -c 0.0 -r 1.0 -m 0.0 -e -d -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs
done

# noise
for noise in $noises
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/noise.csv -c 0.0 -r 1.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs -n $noise
	done
done


# bottleneck size
for bottleneck in $bottlenecks
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/bottleneck.csv -c 0.0 -r 1.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs -b $bottleneck
	done
done

# aggregate results for increased convenience
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/default.csv data/Shapes/ml/experiment_6/aggregated/default.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/decay.csv data/Shapes/ml/experiment_6/aggregated/decay.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/dropout.csv data/Shapes/ml/experiment_6/aggregated/dropout.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/noise.csv data/Shapes/ml/experiment_6/aggregated/noise.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/bottleneck.csv data/Shapes/ml/experiment_6/aggregated/bottleneck.csv


# TODO: grid search on most promising candidates
echo '-b 512 -w 0.0005 -n 0.25' > data/Shapes/ml/experiment_6/grid_search.config
echo '-b 512 -w 0.001 -e -n 0.25' >> data/Shapes/ml/experiment_6/grid_search.config

while IFS= read -r params
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_6/grid_search.csv -c 0.0 -r 1.0 -m 0.0 -f $fold -s 42 --initial_stride 3 --image_size $image_size --noise_only_train --patience $patience --epochs $epochs $walltime $params
	done
done < 'data/Shapes/ml/experiment_6/grid_search.config'

# aggregate results for increased convenience
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_6/grid_search.csv data/Shapes/ml/experiment_6/aggregated/grid_search.csv

