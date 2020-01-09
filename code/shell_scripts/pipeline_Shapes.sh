#!/bin/bash

# no parameter is not allowed
if [ "$#" -ne 1 ]
then
	echo '[ERROR: no argument given, exiting now!]'
	exit 1

# parameter 'paper' means execution of paper code only
elif [ $1 = paper ]
then
	echo '[configuration of CARLA paper]'

	# data analysis setup
	datasets=("visual conceptual")
	aggregators=("mean median")
	image_sizes=("283 100 50 20 10 5")
	dimensions=("FORM LINES")

	# space analysis setup
	tiebreakers=("primary secondary")
	dims=10
	max=5

# parameter 'dissertation' means execution of full code as used in dissertation
elif [ $1 = dissertation ]
then
	echo '[configuration of dissertation]'

	# data analysis setup
	datasets=("visual conceptual")
	aggregators=("mean median")
	image_sizes=("283 100 50 20 10 5")
	dimensions=("FORM LINES")

	# space analysis setup
	tiebreakers=("primary secondary")
	dims=10
	max=5

# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# now execute all individual scripts 
. code/shell_scripts/Shapes/data_analysis.sh
. code/shell_scripts/Shapes/space_analysis.sh
