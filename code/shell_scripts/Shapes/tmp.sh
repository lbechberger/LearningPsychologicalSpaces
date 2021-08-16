#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_aggregators=("mean median")
default_dimension_limit=10
default_visualization_limit=5
default_convexity_limit=5
default_directions=("FORM LINES ORIENTATION visSim artificial")
default_criteria=("kappa spearman")

aggregators="${aggregators:-$default_aggregators}"
dimension_limit="${dimension_limit:-$default_dimension_limit}"
visualization_limit="${visualization_limit:-$default_visualization_limit}"
convexity_limit="${convexity_limit:-$default_convexity_limit}"
directions="${directions:-$default_directions}"
criteria="${criteria:-$default_criteria}"


# analyzing correlation between distances and dissimilarities
# -----------------------------------------------------------

echo 'analyzing correlation between distances and dissimilarities'

echo '    pixel baseline'
for aggregator in $aggregators
do
	echo '        '"$aggregator"
	# if precomputed distances exist: use them; if not: re-compute them
	[ -f 'data/Shapes/mds/analysis/correlations/pixel_distances/283-max-Euclidean.pickle' ] && image_flag='' || image_flag='-i data/Shapes/images/'
	python -u -m code.mds.correlations.pixel_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/analysis/correlations/pixel_distances/' 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/pixel.csv' $image_flag -w 283 -g --kendall -s 42 &> 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/pixel_log.txt' 

	python -m code.mds.correlations.visualize_pixel_correlations 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/pixel.csv' 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/' --kendall &> 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/pixel_best.txt'

done


