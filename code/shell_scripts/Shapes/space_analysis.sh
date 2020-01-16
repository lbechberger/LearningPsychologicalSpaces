#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_aggregators=("mean median")
default_dimension_limit=10
default_visualization_limit=2
default_convexity_limit=5
aggregators="${aggregators:-$default_aggregators}"
dimension_limit="${dimension_limit:-$default_dimension_limit}"
visualization_limit="${visualization_limit:-$default_visualization_limit}"
convexity_limit="${convexity_limit:-$default_convexity_limit}"


# set up the directory structure
echo 'setting up directory structure'
for aggregator in $aggregators
do
	mkdir -p 'data/Shapes/mds/vectors/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/'
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
for aggregator in $aggregators
do
	# create scatter plot for best pixel result
	# TODO: update these values!
	best_pixel_a=mean
	best_pixel_b=42
	best_pixel_d=Euclidean
	python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/best_pixel.png' -p $best_pixel_a -i data/Shapes/images/ -b $best_pixel_b -d $best_pixel_d -g &

	# create scatter plot for best ANN result
	# TODO: update this value!
	best_ann_d=Euclidean
	python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/ann.png' -a /tmp/inception -i data/Shapes/images/ -d $best_ann_d -g &
done
wait

# RQ6: To what extent do the spaces based on median and mean differ?
# ------------------------------------------------------------------

echo 'RQ6: To what extent do the spaces based on median and mean differ?'

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

# analyze convexity
echo '    analyzing convexity'
for aggregator in $aggregators
do
	for i in `seq 1 $convexity_limit`
	do
		python -m code.mds.similarity_spaces.analyze_convexity 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' data/Shapes/raw_data/preprocessed/data_visual.pickle $i -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/convexities.csv' -b -r 100 -s 42 > 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'"$i"'D-convexity.txt' &
	done
done
wait

# analyze interpretable directions
echo '    analyzing interpretable directions'
for aggregator in $aggregators
do
	for dimension in $dimensions
	do
		for i in `seq 1 $dimension_limit`
		do
			# classification
			python -m code.mds.similarity_spaces.analyze_interpretability 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' $i 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/'"$dimension"'-classification.csv' -r 'data/Shapes/mds/regression/'"$dimension"'.pickle' -b -n 100 -s 42 &
			
# regression
			python -m code.mds.similarity_spaces.analyze_interpretability 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' $i 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/'"$dimension"'-regression.csv' -c 'data/Shapes/mds/classification/'"$dimension"'.pickle' -b -n 100 -s 42 &

		done
	done
done
wait


# visualize correlation results
echo '    visualizing correlations'
for aggregator in $aggregators
do
	python -m code.mds.correlations.visualize_correlations -o 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel.csv' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/mds.csv' &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/best.txt' &
done
wait

# visualize MDS spaces
echo '    visualizing MDS spaces'
for aggregator in $aggregators
do
	python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/' -i data/Shapes/images/ -m $visualization_limit &
done
wait
