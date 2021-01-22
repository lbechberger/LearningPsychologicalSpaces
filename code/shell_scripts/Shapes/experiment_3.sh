#!/bin/bash

echo 'experiment 3 - regression on top of sketch classification'

# setting up overall variables
default_folds=("0 1 2 3 4")
folds="${folds:-$default_folds}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	script=code.ml.ann.get_bottleneck_activations
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	script=code/ml/ann/get_bottleneck_activations.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_3/'


# get features of vanilla classifier
for fold in $folds
do
	$cmd $script data/Shapes/ml/dataset/Shapes.pickle 'data/Shapes/ml/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f'"$fold"'_ep0_FINAL.h5' 'data/Shapes/ml/experiment_3/features_vanilla_f'"$fold"'.pickle'
done

# get features of best classification performance

# get features of best correlation to similarity ratings
