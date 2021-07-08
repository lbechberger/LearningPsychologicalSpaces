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
	python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/raw_data/visual_similarities_within.csv 'data/Shapes/raw_data/'"$rating_type"'_similarities.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/individual_ratings.pickle' 'data/Shapes/mds/data_set/individual/similarities/'"$rating_type"'_10.csv' $rating_type $reverse_flag -s between -l -v 10 --seed 42 &> 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/log_preprocessing.txt'

	python -m code.mds.preprocessing.aggregate_similarities 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/individual_ratings.pickle' 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/aggregated_ratings.pickle' 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/' 'data/Shapes/mds/data_set/aggregated/similarities/'"$rating_type"'_median_10.csv' $rating_type --median &> 'data/Shapes/mds/similarities/rating_type/'"$rating_type"'/log_aggregation.txt'

done

python -m code.mds.preprocessing.preprocess_Shapes data/Shapes/raw_data/visual_similarities_within.csv 'data/Shapes/raw_data/visual_similarities.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/similarities/aggregator/individual_ratings.pickle' 'data/Shapes/mds/data_set/individual/similarities/visual_15.csv' visual -s between -l -v 15  --seed 42 &> 'data/Shapes/mds/similarities/aggregator/log_preprocessing.txt'

for aggregator in $aggregators
do
	echo '    visual similarity ('"$aggregator"' aggregation, 15 ratings)'

	[ "$aggregator" == "median" ] && aggregator_flag='--median' || aggregator_flag=''
	python -m code.mds.preprocessing.aggregate_similarities 'data/Shapes/mds/similarities/aggregator/individual_ratings.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/' 'data/Shapes/mds/data_set/aggregated/similarities/visual_'"$aggregator"'_15.csv' visual $aggregator_flag &> 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/log_aggregation.txt'
done

# read in perceptual feature data and preprocess it
for feature in $perceptual_features
do
	echo '    '"$feature"' feature'
	python -m code.mds.preprocessing.preprocess_feature 'data/Shapes/raw_data/'"$feature"'_pre-attentive.csv' 'data/Shapes/raw_data/'"$feature"'_attentive.csv' data/Shapes/raw_data/category_names.csv data/Shapes/raw_data/item_names.csv 'data/Shapes/mds/features/'"$feature"'.pickle' 'data/Shapes/mds/data_set/individual/features/'"$feature"'.csv' 'data/Shapes/mds/data_set/aggregated/features/'"$feature"'.csv' -p 'data/Shapes/mds/visualizations/features/'"$feature"'/' -i data/Shapes/images -q 0.33 &> 'data/Shapes/mds/features/log_'"$feature"'.txt'
done
# dump all of them into common files
python -m code.mds.preprocessing.export_feature_ratings data/Shapes/mds/features data/Shapes/mds/similarities/aggregator/individual_ratings.pickle visual data/Shapes/mds/data_set/individual/features/all_features.csv data/Shapes/mds/data_set/aggregated/features/all_features.csv data/Shapes/mds/data_set/individual/similarities/visual_15_plus_features.csv

# create features from category structure (i.e., 'artificial' and 'visSim') for further downstream analysis
echo '    features based on category structure'
python -m code.mds.preprocessing.features_from_categories data/Shapes/mds/similarities/aggregator/individual_ratings.pickle data/Shapes/mds/features/


# Analysis
# --------
echo 'data analysis'

# comparing psychological features to each other
echo '    comparing psychological features'
for first_feature in $perceptual_features
do
	for second_feature in $perceptual_features
	do
		if [[ "$first_feature" < "$second_feature" ]]
		then
			python -m code.mds.data_analysis.compare_features 'data/Shapes/mds/features/'"$first_feature"'.pickle' 'data/Shapes/mds/features/'"$second_feature"'.pickle' 'data/Shapes/mds/visualizations/features/' -f $first_feature -s $second_feature -i data/Shapes/images
		fi
	done
done

# create average category images at various sizes
echo '    creating average images for all the categories'
for image_size in $image_sizes
do
	echo '        target image size '"$image_size"
	python -m code.mds.data_analysis.average_images data/Shapes/mds/similarities/aggregator/individual_ratings.pickle data/Shapes/images/ -o 'data/Shapes/mds/visualizations/average_images/'"$image_size"'/' -r $image_size &> 'data/Shapes/mds/visualizations/average_images/'"$image_size"'.txt'
done

echo '    comparing visual and conceptual similarity'
python -m  code.mds.data_analysis.find_item_pair_differences data/Shapes/mds/similarities/rating_type/visual/aggregated_ratings.pickle data/Shapes/mds/similarities/rating_type/conceptual/aggregated_ratings.pickle  &> data/Shapes/mds/similarities/rating_type/differences.txt

echo '    visualization of visual vs. conceptual similarity'
python -m code.mds.data_analysis.plot_similarity_matrices data/Shapes/mds/similarities/rating_type/conceptual/aggregated_ratings.pickle data/Shapes/mds/similarities/rating_type/visual/aggregated_ratings.pickle  data/Shapes/mds/visualizations/similarity_matrices/ -f Conceptual -s Visual > data/Shapes/mds/visualizations/similarity_matrices/corr_Conceptual_Visual.txt

echo '    visualization of mean vs. median similarity'
python -m code.mds.data_analysis.plot_similarity_matrices data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/mds/similarities/aggregator/median/aggregated_ratings.pickle  data/Shapes/mds/visualizations/similarity_matrices/ -f Mean -s Median -d > data/Shapes/mds/visualizations/similarity_matrices/corr_Mean_Median.txt


