#!/bin/bash

echo 'experiment 3 - regression on top of sketch classification'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default large small correlation no_noise")
default_noises=("noisy clean")

folds="${folds:-$default_folds}"
regressors="${regressors_ex1:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features:-$default_features}"
noises="${noises:-$default_noises}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	bottleneck_script=code.ml.ann.get_bottleneck_activations
	ann_script=code.ml.ann.run_ann
	regression_script=code.ml.regression.regression
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	bottleneck_script=code/ml/ann/get_bottleneck_activations.sge
	regression_script=code/ml/regression/regression.sge
	ann_script=code/ml/ann/run_ann.sge
	walltime='--walltime 5400'
	qsub ../Utilities/watch_jobs.sge $script ann ../sge-logs/
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi


#for fold in $folds
#do
#	$cmd $ann_script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/noise.csv -c 1.0 -r 0.0 -m 0.0 -e -f $fold -s 42 $walltime --initial_stride 3 --image_size 224 --noise_only_train --patience 200 --epochs 200 -n 0.0
#done

# extract features for both noised and unnoised input
for noise in $noises
do
	if [ $noise = noisy ]
	then
		noise_flag="-n"
	else
		noise_flag=""
	fi

	# define snapshots of classifier trained w/o noise
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f0_ep165_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.tmp
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f1_ep184_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.tmp
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f2_ep169_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.tmp
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f3_ep165_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.tmp
	echo 'data/Shapes/ml/experiment_2/snapshots/c1.0_r0.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.0_mean_4_f4_ep132_FINAL.h5 data/Shapes/ml/experiment_3/features/no_noise_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_3/snapshots.tmp

done


# extract all the features
while IFS= read -r config
do
	$cmd $bottleneck_script data/Shapes/ml/dataset/Shapes.pickle $config -s 42 -i 224
done < 'data/Shapes/ml/experiment_3/snapshots.tmp'

rm data/Shapes/ml/experiment_3/snapshots.tmp



