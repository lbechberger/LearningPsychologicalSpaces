#!/bin/bash

# no parameter is not allowed
if [ "$#" -ne 1 ]
then
	echo '[ERROR: no argument given, exiting now!]'
	exit 1

# parameter 'mds' means execution of similarity space analysis only
elif [ $1 = mds ]
then
	echo '[configuration of MDS paper]'

	# data analysis setup
	rating_types=("visual conceptual")
	aggregators=("mean median")
	image_sizes=("283 100 50 20 10 5")
	perceptual_features=("FORM LINES ORIENTATION")

	# space analysis setup
	dimension_limit=10
	visualization_limit=2	
	convexity_limit=5
	criteria=("kappa spearman")
	directions=("FORM LINES ORIENTATION")

# parameter 'ml' means execution of machine learning experiments only
elif [ $1 = ml ]
then
	echo '[configuration of ML paper]'

	# data set creation
	export_size=256
	image_size=224
	min_size=168
	pickle_noises=("0.0 0.1")
	flip_prob=0.5
	rotation=15
	shear=15

	# experiment 1
	baselines=("--zero")
	regressors=("--linear")
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
	targets=("mean")
	inception_noises=("0.1")
	best_noise=0.1
	comparison_noises=("0.0 0.1")
	shuffled_flag=""
	dims=("")

	# experiment 2
	folds=("0 1 2 3 4")
	weight_decays=("0.0 0.0002 0.001 0.002")
	noises=("0.0 0.25 0.55")
	bottlenecks=("2048 256 128 64 32 16")
	epochs=200
	patience=200

	# experiment 3
	features=("default large small correlation no_noise")
	noises=("noisy clean")

	# experiment 4
	mapping_weights=("0.0625 0.125 0.25 0.5 1 2")
	
	# experiment 5
	ann_config="-e -c 1.0 -r 0.0 -m 0.0625"
	inception_lasso=0.005
	transfer_features="small"
	transfer_lasso=0.02
	

# parameter 'dissertation' means execution of full code as used in dissertation
elif [ $1 = dissertation ]
then
	echo '[configuration of dissertation]'

	# data analysis setup
	rating_types=("visual conceptual")
	aggregators=("mean median")
	image_sizes=("283 100 50 20 10 5")
	perceptual_features=("FORM LINES ORIENTATION")

	# space analysis setup
	dimension_limit=10
	visualization_limit=2
	convexity_limit=5
	criteria=("kappa spearman")
	directions=("FORM LINES ORIENTATION visSim artificial")

	# data set creation
	export_size=256
	image_size=224
	min_size=168
	pickle_noises=("0.0 0.1 0.25 0.55")
	flip_prob=0.5
	rotation=15
	shear=15

	# experiment 1
	baselines=("--zero")
	regressors=("--linear")
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
	targets=("mean median")
	inception_noises=("0.1 0.25 0.55")
	best_noise=0.1
	comparison_noises=("0.0 0.1")
	shuffled_flag="--shuffled"
	dims=("1 2 3 5 6 7 8 9 10")

	# experiment 2
	folds=("0 1 2 3 4")
	weight_decays=("0.0 0.0002 0.001 0.002")
	noises=("0.0 0.25 0.55")
	bottlenecks=("2048 256 128 64 32 16")
	epochs=200
	patience=200

	# experiment 3
	features=("default large small correlation no_noise")
	noises_exp3=("noisy clean")

	# experiment 4
	mapping_weights=("0.0625 0.125 0.25 0.5 1 2")
	
	# experiment 5
	ann_config="-e -c 1.0 -r 0.0 -m 0.0625"
	inception_lasso=0.005
	transfer_features="small"
	transfer_lasso=0.02

	# experiment 6
	weight_decays_enc=("0.0 0.0002 0.001 0.002")
	weight_decays_dec=("0.0002 0.0005 0.001 0.002")
	reconstruction_seeds=("0 42 1337 123456")
	reconstruction_noises=("0.0 0.1 0.25 0.55")

	# experiment 7
	features_exp7=("default reconstruction correlation")
	noises_exp7=("noisy clean")

	# experiment 9
	#TODO update
	ann_config_exp9="-c 0.0 -r 1.0 -m 0.0"
	transfer_features_exp9="reconstruction"
	transfer_lasso_exp9=0.02
	

# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# now execute all individual scripts 
if [ $1 = mds ] || [ $1 = dissertation ]
then
	. code/shell_scripts/Shapes/data_analysis.sh
	. code/shell_scripts/Shapes/space_analysis.sh
fi
if [ $1 = ml ] || [ $1 = dissertation ]
then
	. code/shell_scripts/Shapes/ml_setup.sh
	. code/shell_scripts/Shapes/experiment_1.sh
	. code/shell_scripts/Shapes/experiment_2.sh
	. code/shell_scripts/Shapes/experiment_3.sh
	. code/shell_scripts/Shapes/experiment_4.sh
	. code/shell_scripts/Shapes/experiment_5.sh
fi
if [ $1 = dissertation ]
then
	. code/shell_scripts/Shapes/experiment_6.sh
	. code/shell_scripts/Shapes/experiment_7.sh
	. code/shell_scripts/Shapes/experiment_8.sh
	. code/shell_scripts/Shapes/experiment_9.sh
fi
