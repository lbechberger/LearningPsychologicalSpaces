#!/bin/bash

# no parameter is not allowed
if [ "$#" -ne 1 ]
then
	echo '[ERROR: no argument given, exiting now!]'
	exit 1

# parameter 'paper' means execution of CARLA paper code only
elif [ $1 = paper ]
then
	echo '[configuration of CARLA paper]'

	# MDS setup
	dims=10
	max=5
	algorithms=("metric_SMACOF nonmetric_SMACOF")
	spaces=("metric_SMACOF nonmetric_SMACOF")
	correlation_metrics=("--pearson --spearman")

	# ML setup
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")

	# experiment 1	
	feature_sets_ex1=("ANN pixel_1875 pixel_507")
	lasso_sets_ex1=("ANN pixel_1875 pixel_507")
	baselines_ex1=("--zero")
	regressors_ex1=("--linear")

	# experiment 2
	feature_sets_ex2=("ANN")
	lasso_sets_ex2=("ANN")
	baselines_ex2=("--zero")
	regressors_ex2=("--linear")
	targets_ex2=("metric_SMACOF nonmetric_SMACOF")

	# experiment 3
	feature_sets_ex3=("ANN")
	lasso_sets_ex3=("ANN")
	baselines_ex3=("--zero")
	regressors_ex3=("--linear")
	targets_ex3=("1 2 3 5 6 7 8 9 10")

# parameter 'dissertation' means execution of full code as used in dissertation
elif [ $1 = dissertation ]
then
	echo '[configuration of dissertation]'

	# MDS setup
	dims=10
	max=5
	algorithms=("classical Kruskal metric_SMACOF nonmetric_SMACOF")
	spaces=("classical Kruskal metric_SMACOF nonmetric_SMACOF HorstHout")
	correlation_metrics=("--pearson --spearman --kendall --r2_linear --r2_isotonic")

	# ML setup
	lassos=("0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0")

	# experiment 1	
	feature_sets_ex1=("ANN pixel_1875 pixel_507")
	lasso_sets_ex1=("ANN pixel_1875 pixel_507")
	baselines_ex1=("--zero --mean --normal --draw")
	regressors_ex1=("--linear --random_forest")

	# experiment 2
	feature_sets_ex2=("ANN pixel_1875")
	lasso_sets_ex2=("ANN")
	baselines_ex2=("--zero")
	regressors_ex2=("--linear --random_forest")
	targets_ex2=("classical Kruskal metric_SMACOF nonmetric_SMACOF")

	# experiment 3
	feature_sets_ex3=("ANN")
	lasso_sets_ex3=("ANN")
	baselines_ex3=("--zero")
	regressors_ex3=("--linear")
	targets_ex3=("1 2 3 5 6 7 8 9 10")


# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# now execute all individual scripts 
. code/shell_scripts/NOUN/mds.sh
. code/shell_scripts/NOUN/correlation.sh
. code/shell_scripts/NOUN/ml_setup.sh
. code/shell_scripts/NOUN/experiment_1.sh
. code/shell_scripts/NOUN/experiment_2.sh
. code/shell_scripts/NOUN/experiment_3.sh
