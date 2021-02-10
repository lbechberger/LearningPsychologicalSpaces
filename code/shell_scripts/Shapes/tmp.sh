#!/bin/bash

# setting up overall variables
optimizers=("adam sgd")
paddings=("valid same")
learning_rates=("0.0001 0.001")

qsub ../Utilities/watch_jobs.sge code/ml/ann_run_ann.sge ann ../sge-logs/

for pad in $paddings
do
	for opt in $optimizers
	do
		for lr in $learning_rates
		do
			mkdir -p 'data/Shapes/ml/test/'"$pad"'_'"$opt$"'_'"$lr"'/logs/'
			qsub code/ml/ann/run_ann.sge data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle 'data/Shapes/ml/test/'"$pad"'_'"$opt$"'_'"$lr"'/result.csv' -c 1.0 -r 0.0 -m 0.0 -e -f 0 -s 42 --walltime 5400
		done
	done
done

