#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_aggregators=("mean median")
default_dimension_limit=10
default_visualization_limit=2
default_convexity_limit=5
default_dimensions=("FORM LINES ORIENTATION")

aggregators="${aggregators:-$default_aggregators}"
dimension_limit="${dimension_limit:-$default_dimension_limit}"
visualization_limit="${visualization_limit:-$default_visualization_limit}"
convexity_limit="${convexity_limit:-$default_convexity_limit}"
dimensions="${dimensions:-$default_dimensions}"


# set up the directory structure
echo 'setting up directory structure'
for aggregator in $aggregators
do
	mkdir -p 'data/Shapes/mds/vectors/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/'
done



# RQ5: What amount of information can be gained from the pixels of the image?
# ---------------------------------------------------------------------------

echo 'RQ5: What amount of information can be gained from the pixels of the image?'

for aggregator in $aggregators
do
	echo '    looking at '"$aggregator"' matrix'

	# run pixel baseline
	python -m code.mds.correlations.pixel_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel.csv' -s 283 -g &

	# run ANN baseline
	python -m code.mds.correlations.ann_correlations /tmp/inception 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/ann.csv' &
done
wait

echo '   creating scatter plots of best fits'

# MEAN
# create scatter plot for best pixel result
best_pixel_a=min
best_pixel_b=26
best_pixel_r=11
best_pixel_d=Euclidean
python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/mean/sim.pickle' 'data/Shapes/mds/analysis/aggregator/mean/correlations/best_pixel.png' -p $best_pixel_a -i data/Shapes/images/ -b $best_pixel_b -d $best_pixel_d -g &

# create average category images for best pixel result
python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/mean/' -r $best_pixel_r -a $best_pixel_a &> 'data/Shapes/mds/visualizations/average_images/mean.txt'

# create scatter plot for best ANN result
best_ann_d=Manhattan
python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/mean/sim.pickle' 'data/Shapes/mds/analysis/aggregator/mean/correlations/ann.png' -a /tmp/inception -i data/Shapes/images/ -d $best_ann_d -g &

# MEDIAN
# create scatter plot for best pixel result
best_pixel_a=min
best_pixel_b=24
best_pixel_r=12
best_pixel_d=Euclidean
python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/median/sim.pickle' 'data/Shapes/mds/analysis/aggregator/median/correlations/best_pixel.png' -p $best_pixel_a -i data/Shapes/images/ -b $best_pixel_b -d $best_pixel_d -g &

# create average category images for best pixel result
python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/median/' -r $best_pixel_r -a $best_pixel_a &> 'data/Shapes/mds/visualizations/average_images/median.txt'

# create scatter plot for best ANN result
best_ann_d=Manhattan
python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/median/sim.pickle' 'data/Shapes/mds/analysis/aggregator/median/correlations/ann.png' -a /tmp/inception -i data/Shapes/images/ -d $best_ann_d -g &

wait

# RQ6: How well do the MDS spaces reflect the dissimilarity ratings?
# ------------------------------------------------------------------

echo 'RQ6: How well do the MDS spaces reflect the dissimilarity ratings?'

# run MDS
echo '    running MDS'
for aggregator in $aggregators
do
	Rscript code/mds/similarity_spaces/mds.r -d 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/distance_matrix.csv' -i 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/item_names.csv' -o 'data/Shapes/mds/vectors/'"$aggregator"'/' -n 256 -m 1000 -k $dimension_limit -s 42 --nonmetric_SMACOF -t primary &> 'data/Shapes/mds/vectors/'"$aggregator"'/mds.txt' &
done
wait

# normalize MDS spaces
echo '    normalizing MDS spaces'
for aggregator in $aggregators
do
	python -m code.mds.similarity_spaces.normalize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/' &
done
wait

# do correlation analysis
echo '    correlation of distances in the spaces to dissimilarities from the matrices'
for source_aggregator in $aggregators
do
	for target_aggregator in $aggregators
	do
		python -m code.mds.correlations.mds_correlations 'data/Shapes/mds/similarities/aggregator/'"$target_aggregator"'/sim.pickle' 'data/Shapes/mds/vectors/'"$source_aggregator"'/' -o 'data/Shapes/mds/analysis/aggregator/'"$source_aggregator"'/correlations/mds_to_'"$target_aggregator"'.csv' --n_max $dimension_limit &
	done
done
wait

echo '    correlation of distances on the interpretable dimensions to dissimilarities from the matrices'
for aggregator in $aggregators
do
	python -m code.mds.correlations.dimension_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/regression/' -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/dims.csv' &
done

# RQ7: How well do the MDS Spaces Enforce the Convexity of Conceptual Regions?
# ----------------------------------------------------------------------------

echo 'RQ7: How well do the MDS Spaces Enforce the Convexity of Conceptual Regions?'

for aggregator in $aggregators
do
	for i in `seq 1 $convexity_limit`
	do
		python -m code.mds.similarity_spaces.analyze_convexity 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' data/Shapes/raw_data/preprocessed/data_visual.pickle $i -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/convexities.csv' -b -r 100 -s 42 > 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'"$i"'D-convexity.txt' &
		python -m code.mds.similarity_spaces.analyze_density 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv'  data/Shapes/raw_data/preprocessed/data_visual.pickle $i -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/densities.csv'-b -r 100 -s 42 > 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'"$i"'D-density.txt' &
	done
done
wait

# RQ8: Are the interpretable directions reflected in the MDS spaces?
# ------------------------------------------------------------------

echo 'RQ8: Are the interpretable directions reflected in the MDS spaces?'

echo '    finding directions'
for aggregator in $aggregators
do
	for dimension in $dimensions
	do
		for i in `seq 1 $dimension_limit`
		do
			python -m code.mds.similarity_spaces.find_directions 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' $i 'data/Shapes/mds/classification/'"$dimension"'.pickle' 'data/Shapes/mds/regression/'"$dimension"'.pickle' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/'"$dimension"'.csv' &
		done
	done
done
wait

echo '    comparing directions'
for aggregator in $aggregators
do
	python -m code.mds.similarity_spaces.compare_directions 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/' $dimension_limit 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/similarities.csv' &
done
wait

#TODO compare cosine similarity of directions

#TODO filter interpretable dimensions




# Some additional visualizations
# ------------------------------

echo 'Some additional visualizations'

# visualize correlation results
echo '    visualizing correlations'
for aggregator in $aggregators
do
	python -m code.mds.correlations.visualize_correlations -o 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel.csv' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/mds_to_'"$aggregator"'.csv' &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/best.txt' &
done
wait

# visualize MDS spaces
echo '    visualizing MDS spaces'
for aggregator in $aggregators
do
	python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/' -i data/Shapes/images/ -m $visualization_limit &
done
wait
