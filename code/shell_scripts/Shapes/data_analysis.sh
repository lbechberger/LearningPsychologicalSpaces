#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_datasets=("visual conceptual")
default_aggregators=("mean median")
default_image_sizes=("283 100 50 20 10 5")

datasets="${datasets:-$default_datasets}"
aggregators="${aggregators:-$default_aggregators}"
image_sizes="${image_sizes:-$default_image_sizes}"

# set up the directory structure
echo 'setting up directory structure'
for dataset in $datasets
do
	mkdir -p 'data/Shapes/mds/analysis/dataset/'"$dataset"'/'
	mkdir -p 'data/Shapes/mds/similarities/dataset/'"$dataset"'/'
done		

for aggregator in $aggregators
do
		mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/'
		mkdir -p 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/'
done

for image_size in $image_sizes
do
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/'
done

# Preprocessing
# -------------

echo 'preprocessing data'

# read in similarity data and preprocess it
for dataset in $datasets
do
	echo '    reading CSV files for '"$dataset"' similarity'
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/mds/raw_data/within.csv 'data/Shapes/mds/raw_data/'"$dataset"'.csv' 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' &> 'data/Shapes/mds/raw_data/preprocess_'"$dataset"'.txt'

done

# TODO: read in dimension data and preprocess it

	
	
# RQ1: Comparing conceptual to visual similarity
# ----------------------------------------------

echo 'RQ1: comparting conceptual and visual similarity matrices (median only)'
echo '    aggregating ratings'
for dataset in $datasets
do
	# use a limit of 10, because conceptual similarity has only 10 ratings per pair
	python -m code.mds.preprocessing.compute_similarities 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' 'data/Shapes/mds/similarities/dataset/'"$dataset"'/sim.pickle' -s between -l -v 10 -p --median &> 'data/Shapes/mds/similarities/dataset/'"$dataset"'/log.txt'
done

echo '    correlations'
python -m code.mds.correlations.visual_conceptual_correlations 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' -o 'data/Shapes/mds/analysis/dataset/' -p &> 'data/Shapes/mds/analysis/dataset/correlations.txt'
	
echo '    differences'
python -m  code.mds.preprocessing.compare_visual_conceptual 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' &> 'data/Shapes/mds/analysis/datset/differences.txt'
done

# RQ2: Do 'Sim' categories have higher internal shape similarity than 'Dis' categories?
# -------------------------------------------------------------------------------------

# TODO
# --> analyze_similarity_distribution
#python -m code.mds.preprocessing.analyze_similarity_distribution 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' -s between -o 'data/Shapes/mds/analysis/'"$dataset"'/'"$aggregator"'/' $aggregator_flag &> 'data/Shapes/mds/analysis/'"$dataset"'/'"$aggregator"'/analysis.txt'

# create average images of the categories
echo 'creating average images for all the categories'
for image_size in $image_sizes
do
	echo '    target image size '"$image_size"
	python -m code.mds.preprocessing.average_images data/Shapes/mds/raw_data/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/' -r $image_size &> 'data/Shapes/mds/visualizations/average_images/'"$image_size"'.txt'
done

# RQ3: Comparing binary to continuous dimension ratings
# -----------------------------------------------------

# TODO

# RQ4: Comparing dissimilarity matrices of median aggregation and mean aggregation
# --------------------------------------------------------------------------------

echo 'RQ4: aggregation with median vs aggregation with mean'
echo '    aggregating similarities'
for aggregator in $aggregators
do
	echo '        computing average similarities with '"$aggregator"
	[ "$aggregator" == "median" ] && aggregator_flag='--median' || aggregator_flag=''

	# use a limit of 15 because we have more data for the visual similarities
	python -m code.mds.preprocessing.compute_similarities 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' -s between -l -v 15 -p $aggregator_flag &> 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/log.txt'

	echo '        creating CSV files for MDS'
	python -m code.mds.preprocessing.pickle_to_csv 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/'

# TODO: Spearman correlation and scatter plot --> visual_conceptual_correlations?
# TODO: two histograms (rough and fine)	--> compute_similarities
# TODO: count #unique values --> compute_similarities

