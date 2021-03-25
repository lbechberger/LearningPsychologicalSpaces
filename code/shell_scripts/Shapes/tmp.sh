#!/bin/bash

echo 'experiment 3 - regression on top of sketch classification'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default large small correlation")
default_noises=("noisy clean")

folds="${folds:-$default_folds}"
regressors="${regressors_ex1:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features:-$default_features}"
noises="${noises:-$default_noises}"

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
done

