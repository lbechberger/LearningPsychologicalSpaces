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

# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# now execute all individual scripts 
. code/shell_scripts/Shapes/data_analysis.sh
. code/shell_scripts/Shapes/space_analysis.sh
