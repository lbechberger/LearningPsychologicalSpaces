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

	baselines_ex1=("--zero")
	regressors_ex1=("--linear")
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
	noises=("0.1 0.25 0.55")
	best_noise=0.1
	dims=("1 2 3 5 6 7 8 9 10")


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
	visualization_limit=5
	convexity_limit=5
	criteria=("kappa spearman")
	directions=("FORM LINES ORIENTATION visSim artificial")

	# machine learning setup
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")
	noises=("0.1 0.25 0.55")

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
fi
