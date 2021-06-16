echo 'experiment 7 - regression on top of autoencoder'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_regressors=("--linear")
default_lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
default_features=("default best")
default_noises=("noisy clean")
default_image_size=224

folds="${folds:-$default_folds}"
regressors="${regressors:-$default_regressors}"
lassos="${lassos:-$default_lassos}"
features="${features_exp7:-$default_features}"
noises="${noises_exp7:-$default_noises}"
image_size="${image_size:-$default_image_size}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	bottleneck_script=code.ml.ann.get_bottleneck_activations
	regression_script=code.ml.regression.regression
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	bottleneck_script=code/ml/ann/get_bottleneck_activations.sge
	regression_script=code/ml/regression/regression.sge
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_7/features' 'data/Shapes/ml/experiment_7/aggregated'

# extract features for both noised and unnoised input
for noise in $noises
do
	if [ $noise = noisy ]
	then
		noise_flag="-n 0.1"
	else
		noise_flag=""
	fi

	# define snapshots of default autoencoder
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f0_ep95_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f1_ep25_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep48_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f3_ep56_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f4_ep52_FINAL.h5 data/Shapes/ml/experiment_7/features/default_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config


	# define snapshots of best configuration
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f0_ep195_FINAL.h5 data/Shapes/ml/experiment_7/features/best_f0_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f1_ep192_FINAL.h5 data/Shapes/ml/experiment_7/features/best_f1_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f2_ep175_FINAL.h5 data/Shapes/ml/experiment_7/features/best_f2_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f3_ep196_FINAL.h5 data/Shapes/ml/experiment_7/features/best_f3_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config
	echo 'data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f4_ep199_FINAL.h5 data/Shapes/ml/experiment_7/features/best_f4_'$noise$'.pickle '"$noise_flag" >> data/Shapes/ml/experiment_7/snapshots.config

done

# extract all the features
while IFS= read -r config
do
	$cmd $bottleneck_script data/Shapes/ml/dataset/Shapes.pickle $config -s 42 -i $image_size
done < 'data/Shapes/ml/experiment_7/snapshots.config'

