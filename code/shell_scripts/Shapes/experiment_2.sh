#!/bin/bash

echo 'experiment 2 - classification baseline'

default_folds=("0 1 2 3 4")
folds="${folds:-$default_folds}"

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_2/logs/'

for fold in $folds
do
	python -m code.ml.ann.run_ann data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/output.csv -c 0.25 -r 0.5 -m 0.25 -e -t -f $fold -s 42
done

python -m code.ml.ann.get_bottleneck_activations data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/snapshots/c0.25_r0.5_m0.25_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep0_FINAL.h5 data/Shapes/ml/experiment_2/features.pickle

