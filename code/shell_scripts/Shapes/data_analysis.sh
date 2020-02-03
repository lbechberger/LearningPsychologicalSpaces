#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_datasets=("visual conceptual")
default_aggregators=("mean median")
default_image_sizes=("283 100 50 20 10 5")
default_dimensions=("FORM LINES ORIENTATION")

datasets="${datasets:-$default_datasets}"
aggregators="${aggregators:-$default_aggregators}"
image_sizes="${image_sizes:-$default_image_sizes}"
dimensions="${dimensions:-$default_dimensions}"

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

for dimension in $dimensions
do
	mkdir -p 'data/Shapes/mds/analysis/dimension/'"$dimension"'/'
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

# read in dimension data and preprocess it
for dimension in $dimensions
do
	echo '    reading CSV files for '"$dimension"' ratings'
	python -m code.mds.preprocessing.preprocess_dimension 'data/Shapes/raw_data/'"$dimension"'_binary.csv' 'data/Shapes/raw_data/'"$dimension"'_continuous.csv' 'data/Shapes/raw_data/preprocessed/'"$dimension"'.pickle' -p -o 'data/Shapes/mds/analysis/dimension/'"$dimension"'/' &> 'data/Shapes/raw_data/preprocessed/preprocess_'"$dimension"'.txt'

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

# RQ3: Comparing binary to continuous dimension ratings
# -----------------------------------------------------

echo 'RQ3: comparing binary to continuous dimension ratings'

# analyze each dimension and construct regression & classification problem
for dimension in $dimensions
do
	echo '    looking at '"$dimension"' data'
	python -m code.mds.preprocessing.analyze_dimension 'data/Shapes/raw_data/preprocessed/'"$dimension"'.pickle' 'data/Shapes/mds/analysis/dimension/'"$dimension"'/' 'data/Shapes/mds/classification/'"$dimension"'.pickle' 'data/Shapes/mds/regression/'"$dimension"'.pickle' -i data/Shapes/images &> 'data/Shapes/mds/analysis/dimension/'"$dimension"'/analysis.txt'
done

# compare dimensions pairwise
python -m code.mds.preprocessing.compare_dimensions 'data/Shapes/mds/regression/FORM.pickle' 'data/Shapes/mds/regression/LINES.pickle' 'data/Shapes/mds/analysis/dimension/' -f FORM -s LINES -i data/Shapes/images &> 'data/Shapes/mds/analysis/dimension/FORM-LINES.txt'
python -m code.mds.preprocessing.compare_dimensions 'data/Shapes/mds/regression/FORM.pickle' 'data/Shapes/mds/regression/ORIENTATION.pickle' 'data/Shapes/mds/analysis/dimension/' -f FORM -s ORIENTATION -i data/Shapes/images &> 'data/Shapes/mds/analysis/dimension/FORM-ORIENTATION.txt'
python -m code.mds.preprocessing.compare_dimensions 'data/Shapes/mds/regression/LINES.pickle' 'data/Shapes/mds/regression/ORIENTATION.pickle' 'data/Shapes/mds/analysis/dimension/' -f LINES -s ORIENTATION -i data/Shapes/images &> 'data/Shapes/mds/analysis/dimension/LINES-ORIENTATION.txt'

# create dimensions from category structure
python -m code.mds.preprocessing.dimensions_from_categories data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/mds/regression/ data/Shapes/mds/classification -s between


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

