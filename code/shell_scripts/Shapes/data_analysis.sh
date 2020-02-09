#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_datasets=("visual conceptual")
default_aggregators=("mean median")
default_image_sizes=("283 100 50 20 10 5")
default_perceptual_features=("FORM LINES ORIENTATION")

datasets="${datasets:-$default_datasets}"
aggregators="${aggregators:-$default_aggregators}"
image_sizes="${image_sizes:-$default_image_sizes}"
perceptual_features="${perceptual_features:-$default_perceptual_features}"

# set up the directory structure
echo 'setting up directory structure'

mkdir -p data/Shapes/raw_data/preprocessed
mkdir -p data/Shapes/mds/classification
mkdir -p data/Shapes/mds/regression
mkdir -p data/Shapes/mds/analysis/aggregator

for dataset in $datasets
do
	mkdir -p 'data/Shapes/mds/analysis/dataset/'"$dataset"'/'
	mkdir -p 'data/Shapes/mds/similarities/dataset/'"$dataset"'/'
done		

for aggregator in $aggregators
do
	mkdir -p 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/'
done

for feature in $perceptual_features
do
	mkdir -p 'data/Shapes/mds/analysis/features/'"$feature"'/'
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
	[ "$dataset" == "conceptual" ] && reverse_flag='--reverse' || reverse_flag=''	
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/raw_data/within.csv 'data/Shapes/raw_data/'"$dataset"'.csv' 'data/Shapes/raw_data/preprocessed/data_'"$dataset"'.pickle' $reverse_flag &> 'data/Shapes/raw_data/preprocessed/preprocess_'"$dataset"'.txt'

done

# read in perceptual feature data and preprocess it
for feature in $perceptual_features
do
	echo '    reading CSV files for '"$feature"' ratings'
	python -m code.mds.preprocessing.preprocess_feature 'data/Shapes/raw_data/'"$feature"'_binary.csv' 'data/Shapes/raw_data/'"$feature"'_continuous.csv' 'data/Shapes/raw_data/preprocessed/'"$feature"'.pickle' -p -o 'data/Shapes/mds/analysis/features/'"$feature"'/' &> 'data/Shapes/raw_data/preprocessed/preprocess_'"$feature"'.txt'

done
	
	
# RQ1: Comparing conceptual to visual similarity
# ----------------------------------------------

echo 'RQ1: comparing conceptual and visual similarity matrices (median only)'
echo '    aggregating ratings'
for dataset in $datasets
do
	# use a limit of 10, because conceptual similarity has only 10 ratings per pair
	python -m code.mds.preprocessing.compute_similarities 'data/Shapes/raw_data/preprocessed/data_'"$dataset"'.pickle' 'data/Shapes/mds/similarities/dataset/'"$dataset"'/sim.pickle' -s between -l -v 10 -p --median &> 'data/Shapes/mds/similarities/dataset/'"$dataset"'/log.txt'
done

echo '    correlations'
# full matrices
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' -o 'data/Shapes/mds/analysis/dataset/' -p -f 'visual' -s 'conceptual' &> 'data/Shapes/mds/analysis/dataset/correlations.txt'
# only 'Sim' categories
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' -f 'visual(Sim)' -s 'conceptual(Sim)' --sim_only 'data/Shapes/raw_data/preprocessed/data_visual.pickle' &> 'data/Shapes/mds/analysis/dataset/correlations(Sim).txt'

	
echo '    differences'
python -m  code.mds.preprocessing.compare_visual_conceptual 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' &> 'data/Shapes/mds/analysis/dataset/differences.txt'


# RQ2: Do 'Sim' categories have higher internal shape similarity than 'Dis' categories?
# -------------------------------------------------------------------------------------

echo 'RQ2: Does the Sim-Dis distinction reflect visual similarity?'

echo '    analyzing raw data'
for dataset in $datasets
do
	echo '        '"$dataset"
	python -m code.mds.preprocessing.analyze_similarity_distribution 'data/Shapes/raw_data/preprocessed/data_'"$dataset"'.pickle' -s between --median &> 'data/Shapes/mds/analysis/dataset/'"$dataset"'/analysis.txt'
done

echo '    creating average images for all the categories'
for image_size in $image_sizes
do
	echo '        target image size '"$image_size"
	python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/' -r $image_size &> 'data/Shapes/mds/visualizations/average_images/'"$image_size"'.txt'
done

# RQ3: Comparing pre-attentive to attentive ratings of perceptual features
# ------------------------------------------------------------------------

echo 'RQ3: Comparing pre-attentive to attentive ratings of perceptual features'

# analyze each perceptual feature and construct regression & classification problem
for feature in $perceptual_features
do
	echo '    looking at '"$feature"' data'
	python -m code.mds.preprocessing.analyze_feature 'data/Shapes/raw_data/preprocessed/'"$feature"'.pickle' 'data/Shapes/mds/analysis/features/'"$feature"'/' 'data/Shapes/mds/classification/'"$feature"'.pickle' 'data/Shapes/mds/regression/'"$feature"'.pickle' -i data/Shapes/images &> 'data/Shapes/mds/analysis/features/'"$feature"'/analysis.txt'
done

# compare features pairwise
for first_feature in $perceptual_features
do
	for second_feature in $perceptual_features
	do
		if [[ "$first_feature" < "$second_feature" ]]
		then
			python -m code.mds.preprocessing.compare_features 'data/Shapes/mds/regression/'"$first_feature"'.pickle' 'data/Shapes/mds/regression/'"$second_feature"'.pickle' 'data/Shapes/mds/analysis/features/' -f $first_feature -s $second_feature -i data/Shapes/images &> 'data/Shapes/mds/analysis/features/'"$first_feature"'-'$second_feature'.txt'
		fi
	done
done

# create features from category structure (i.e., 'artificial' and 'visSim')
python -m code.mds.preprocessing.features_from_categories data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/mds/regression/ data/Shapes/mds/classification -s between


# RQ4: Comparing dissimilarity matrices of median aggregation and mean aggregation
# --------------------------------------------------------------------------------

echo 'RQ4: aggregation with median vs aggregation with mean'
echo '    aggregating similarities'
for aggregator in $aggregators
do
	echo '        computing average similarities with '"$aggregator"
	[ "$aggregator" == "median" ] && aggregator_flag='--median' || aggregator_flag=''

	# use a limit of 15 because we have more data for the visual similarities
	python -m code.mds.preprocessing.compute_similarities 'data/Shapes/raw_data/preprocessed/data_visual.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' -s between -l -v 15 -p $aggregator_flag &> 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/log.txt'

	echo '        creating CSV files for MDS'
	python -m code.mds.preprocessing.pickle_to_csv 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/'
done

echo '    computing correlation of the aggregated similarity ratings'
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/aggregator/median/sim.pickle' 'data/Shapes/mds/similarities/aggregator/mean/sim.pickle' -o 'data/Shapes/mds/analysis/aggregator/' -p -f 'median' -s 'mean' &> 'data/Shapes/mds/analysis/aggregator/correlations.txt'

