#!/bin/bash

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
	for aggregator in $aggregators
	do
		mkdir -p 'data/Shapes/mds/analysis/'"$dataset"'/'"$aggregator"'/'
		mkdir -p 'data/Shapes/mds/similarities/'"$dataset"'/'"$aggregator"'/'
	done
done
for image_size in $image_sizes
do
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/'
done


# read in similarity data and preprocess it
echo 'preprocessing data'
for dataset in $datasets
do
	echo '    reading CSV files for '"$dataset"' similarity'
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/mds/raw_data/within.csv 'data/Shapes/mds/raw_data/'"$dataset"'.csv' 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' &> 'data/Shapes/mds/raw_data/preprocess_'"$dataset"'.txt'

	for aggregator in $aggregators
	do
		echo '        computing average similarities with '"$aggregator"
		[ "$aggregator" == "median" ] && aggregator_flag='--median' || aggregator_flag=''
		python -m code.mds.preprocessing.compute_similarities 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' 'data/Shapes/mds/similarities/'"$dataset"'/'"$aggregator"'/sim.pickle' -s between -l -p $aggregator_flag &> 'data/Shapes/mds/similarities/'"$dataset"'/'"$aggregator"'/log.txt'

		echo '        creating CSV files for MDS'
		python -m code.mds.preprocessing.pickle_to_csv 'data/Shapes/mds/similarities/'"$dataset"'/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/similarities/'"$dataset"'/'"$aggregator"'/'

		echo '        analyzing the distribution of similarity ratings'
		python -m code.mds.preprocessing.analyze_similarities 'data/Shapes/mds/raw_data/data_'"$dataset"'.pickle' -s between -o 'data/Shapes/mds/analysis/'"$dataset"'/'"$aggregator"'/' $aggregator_flag &> 'data/Shapes/mds/analysis/'"$dataset"'/'"$aggregator"'/analysis.txt'
	
	done

done

# compare conceptual to visual similarity
echo 'comparting conceptual and visual similarity matrices'
for aggregator in $aggregators
do
	echo '    correlations for '"$aggregator"
	python -m code.mds.correlations.visual_conceptual_correlations 'data/Shapes/mds/similarities/visual/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/similarities/conceptual/'"$aggregator"'/sim.pickle' -o 'data/Shapes/mds/analysis/conceptual/'"$aggregator" -p &> 'data/Shapes/mds/analysis/conceptual/'"$aggregator"'/correlations.txt'
	
	echo '    differences for '"$aggregator"
	python -m  code.mds.preprocessing.compare_visual_conceptual 'data/Shapes/mds/similarities/visual/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/similarities/conceptual/'"$aggregator"'/sim.pickle' &> 'data/Shapes/mds/analysis/conceptual/'"$aggregator"'/differences.txt'
done


# create average images of the categories
echo 'creating average images for all the categories'
for image_size in $image_sizes
do
	echo '    target image size '"$image_size"
	python -m code.mds.preprocessing.average_images data/Shapes/mds/raw_data/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/' -r $image_size &> 'data/Shapes/mds/visualizations/average_images/'"$image_size"'.txt'
done

# TODO: read in data about dimensions and analyze it




