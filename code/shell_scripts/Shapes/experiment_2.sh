#!/bin/bash

echo 'experiment 2 - classification baseline'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_weight_decays=("0.0 0.0002 0.001 0.002")
default_noises=("0.25 0.55")
default_bottlenecks=("256 128 64 32 16")

folds="${folds:-$default_folds}"
weight_decays="${weight_decays:-$default_weight_decays}"
noises="${noises:-$default_noises}"
bottlenecks="${bottlenecks:-$default_bottlenecks}"

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
mkdir -p 'data/Shapes/ml/experiment_2/logs/' 'data/Shapes/ml/experiment_2/aggregated'

# vanilla setup
for fold in $folds
do
	$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/vanilla.csv -c 1.0 -r 0.0 -m 0.0 -e -f $fold -s 42 $walltime
done

# weight decay
for weight_decay in $weight_decays
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/decay.csv -c 1.0 -r 0.0 -m 0.0 -e -f $fold -s 42 $walltime -w $weight_decay
	done
done

# no dropout
for fold in $folds
do
	$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/dropout.csv -c 1.0 -r 0.0 -m 0.0 -f $fold -s 42 $walltime
done

# noise
for noise in $noises
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/noise.csv -c 1.0 -r 0.0 -m 0.0 -e -f $fold -s 42 $walltime -n $noise
	done
done


# bottleneck size
for bottleneck in $bottlenecks
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/bottleneck.csv -c 1.0 -r 0.0 -m 0.0 -e -f $fold -s 42 $walltime -b $bottleneck
	done
done


# aggregate results for increased convenience
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_2/vanilla.csv data/Shapes/ml/experiment_2/aggregated/vanilla.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_2/decay.csv data/Shapes/ml/experiment_2/aggregated/decay.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_2/dropout.csv data/Shapes/ml/experiment_2/aggregated/dropout.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_2/noise.csv data/Shapes/ml/experiment_2/aggregated/noise.csv
python -m code.ml.ann.average_folds data/Shapes/ml/experiment_2/bottleneck.csv data/Shapes/ml/experiment_2/aggregated/bottleneck.csv

