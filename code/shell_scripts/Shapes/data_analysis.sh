#!/bin/bash

# Overall Setup
# -------------

# set up global variables
default_rating_types=("visual conceptual")
default_aggregators=("mean median")
default_image_sizes=("283 100 50 20 10 5")
default_perceptual_features=("FORM LINES ORIENTATION")

rating_types="${rating_types:-$default_rating_types}"
aggregators="${aggregators:-$default_aggregators}"
image_sizes="${image_sizes:-$default_image_sizes}"
perceptual_features="${perceptual_features:-$default_perceptual_features}"

# set up the directory structure
echo 'setting up directory structure'

mkdir -p data/Shapes/mds/features 
mkdir -p data/Shapes/mds/data_set/individual/features data/Shapes/mds/data_set/individual/similarities
mkdir -p data/Shapes/mds/data_set/aggregated/features data/Shapes/mds/data_set/aggregated/similarities
mkdir -p data/Shapes/mds/visualizations/similarity_matrices

for rating_type in $rating_types
do
	mkdir -p 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/'
done		

for aggregator in $aggregators
do
	mkdir -p 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/'
done

for image_size in $image_sizes
do
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/'
done

for feature in $perceptual_features
do
	mkdir -p 'data/Shapes/mds/visualizations/features/'"$feature"'/'
done

# Preprocessing
# -------------

echo 'preprocessing data'

# read in similarity data and preprocess it
for rating_type in $rating_types
do
	echo '    '"$rating_type"' similarity (median aggregation, 10 ratings)'
	[ "$rating_type" == "conceptual" ] && reverse_flag='--reverse' || reverse_flag=''
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/raw_data/visual_similarities_within.csv 'data/Shapes/raw_data/'"$rating_type"'_similarities.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/individual_ratings.pickle' 'data/Shapes/mds/data_set/individual/similarities/'"$rating_type"'_10.csv' $rating_type $reverse_flag -s between -l -v 10 &> 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/log_preprocessing.txt'

	python -m code.mds.preprocessing.aggregate_similarities 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/individual_ratings.pickle' 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/aggregated_ratings.pickle' 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/' 'data/Shapes/mds/data_set/aggregated/similarities/'"$rating_type"'_median_10.csv' $rating_type --median &> 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/log_aggregation.txt'

done

for aggregator in $aggregators
do
	echo '    visual similarity ('"$aggregator"' aggregation, 15 ratings)'
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/raw_data/visual_similarities_within.csv 'data/Shapes/raw_data/visual_similarities.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/similarities/aggregator/individual_ratings.pickle' 'data/Shapes/mds/data_set/individual/similarities/visual_15.csv' visual -s between -l -v 15 &> 'data/Shapes/mds/similarities/aggregator/log_preprocessing.txt'

	[ "$aggregator" == "median" ] && aggregator_flag='--median' || aggregator_flag=''
	python -m code.mds.preprocessing.aggregate_similarities 'data/Shapes/mds/similarities/aggregator/individual_ratings.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/' 'data/Shapes/mds/data_set/aggregated/similarities/visual_'"$aggregator"'_15.csv' visual $aggregator_flag &> 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/log_aggregation.txt'
done

# read in perceptual feature data and preprocess it
for feature in $perceptual_features
do
	echo '    '"$feature"' feature'
	python -m code.mds.preprocessing.preprocess_feature 'data/Shapes/raw_data/'"$feature"'_pre-attentive.csv' 'data/Shapes/raw_data/'"$feature"'_attentive.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/features/'"$feature"'.pickle' 'data/Shapes/mds/data_set/individual/features/'"$feature"'.csv' 'data/Shapes/mds/data_set/aggregated/features/'"$feature"'.csv' -p 'data/Shapes/mds/visualizations/features/'"$feature"'/' -i data/Shapes/images &> 'data/Shapes/mds/features/log_'"$feature"'.txt'
done
# dump all of them into common files
python -m code.mds.preprocessing.export_feature_ratings data/Shapes/mds/features data/Shapes/mds/data_set/individual/features/all_features.csv data/Shapes/mds/data_set/aggregated/features/all_features.csv



	
	
# RQ1: Comparing conceptual to visual similarity
# ----------------------------------------------

echo 'RQ1: comparing conceptual and visual similarity matrices (median only)'
echo '    aggregating rating_types'
for rating_type in $rating_types
do
	# use a limit of 10, because conceptual similarity has only 10 rating_types per pair
	python -m code.mds.preprocessing.compute_similarities 'data/Shapes/raw_data/preprocessed/data_'"$rating_type"'.pickle' 'data/Shapes/mds/similarities/dataset/'"$rating_type"'/sim.pickle' -s between -l -v 10 -p --median &> 'data/Shapes/mds/similarities/dataset/'"$rating_type"'/log.txt'

	# output the matrices in csv style for easier inspection
	python -m code.mds.preprocessing.pickle_to_csv 'data/Shapes/mds/similarities/dataset/'"$rating_type"'/sim.pickle' 'data/Shapes/mds/similarities/dataset/'"$rating_type"'/'
done


echo '    correlations'
# full matrices
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' -o 'data/Shapes/mds/analysis/dataset/' -p -f 'Visual' -s 'Conceptual' &> 'data/Shapes/mds/analysis/dataset/correlations.txt'
# only 'Sim' categories
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' -f 'Visual (Sim)' -s 'Conceptual (Sim)' --sim_only 'data/Shapes/raw_data/preprocessed/data_visual.pickle' &> 'data/Shapes/mds/analysis/dataset/correlations(Sim).txt'

	
echo '    differences'
python -m  code.mds.preprocessing.compare_visual_conceptual 'data/Shapes/mds/similarities/dataset/visual/sim.pickle' 'data/Shapes/mds/similarities/dataset/conceptual/sim.pickle' &> 'data/Shapes/mds/analysis/dataset/differences.txt'

echo '    visualization'
python -m code.mds.preprocessing.plot_similarity_tables data/Shapes/mds/similarities/dataset/visual/sim.pickle data/Shapes/mds/similarities/dataset/conceptual/sim.pickle data/Shapes/mds/visualizations/similarities 


# RQ2: Do 'Sim' categories have higher internal shape similarity than 'Dis' categories?
# -------------------------------------------------------------------------------------

echo 'RQ2: Does the Sim-Dis distinction reflect visual similarity?'

echo '    analyzing raw data'
for rating_type in $rating_types
do
	echo '        '"$rating_type"
	python -m code.mds.preprocessing.analyze_similarity_distribution 'data/Shapes/raw_data/preprocessed/data_'"$rating_type"'.pickle' -s between --median &> 'data/Shapes/mds/analysis/dataset/'"$rating_type"'/analysis.txt'
done

echo '    creating average images for all the categories'
for image_size in $image_sizes
do
	echo '        target image size '"$image_size"
	python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/' -r $image_size &> 'data/Shapes/mds/visualizations/average_images/'"$image_size"'.txt'
done

# RQ3: Comparing pre-attentive to attentive rating_types of perceptual features
# ------------------------------------------------------------------------

echo 'RQ3: Comparing pre-attentive to attentive rating_types of perceptual features'

# analyze each perceptual feature and construct regression & classification problem
for feature in $perceptual_features
do
	echo '    looking at '"$feature"' data'
	python -m code.mds.preprocessing.analyze_feature 'data/Shapes/raw_data/preprocessed/'"$feature"'.pickle' 'data/Shapes/mds/analysis/features/'"$feature"'/' 'data/Shapes/mds/classification/'"$feature"'.pickle' 'data/Shapes/mds/regression/'"$feature"'.pickle' -i data/Shapes/images -m &> 'data/Shapes/mds/analysis/features/'"$feature"'/analysis.txt'
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

# create features from category structure (i.e., 'artificial' and 'visSim') for further downstream analysis
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

echo '    computing correlation of the aggregated similarity rating_types'
python -m code.mds.correlations.similarity_correlations 'data/Shapes/mds/similarities/aggregator/median/sim.pickle' 'data/Shapes/mds/similarities/aggregator/mean/sim.pickle' -o 'data/Shapes/mds/analysis/aggregator/' -p -f 'Median' -s 'Mean' &> 'data/Shapes/mds/analysis/aggregator/correlations.txt'

