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


# set up the directory structure
echo 'setting up directory structure'
for aggregator in $aggregators
do
	mkdir -p 'data/Shapes/mds/data_set/spaces/coordinates/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/data_set/spaces/directions/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/clean/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/regions/'
	for criterion in $criteria
	do
		mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/directions/'"$criterion"'/'
		mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/full/'"$criterion"'/'
	done
	mkdir -p 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/shepard/'
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/regions/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/directions/'"$aggregator"'/aggregated/'
done
mkdir -p 'data/Shapes/mds/analysis/correlations/pixel_distances/'

# create and normalize similarity spaces
# --------------------------------------
echo 'creating similarity spaces'
echo '    running MDS'
for aggregator in $aggregators
do
	Rscript code/mds/similarity_spaces/mds.r -d 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/distance_matrix.csv' -i 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/item_names.csv' -o 'data/Shapes/mds/data_set/spaces/coordinates/'"$aggregator"'/' -n 256 -m 1000 -k $dimension_limit -s 42 --nonmetric_SMACOF -t primary &> 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/mds.txt' &
done
wait

echo '    normalizing MDS spaces'
for aggregator in $aggregators
do
	python -m code.mds.similarity_spaces.normalize_spaces 'data/Shapes/mds/data_set/spaces/coordinates/'"$aggregator"'/' data/Shapes/mds/similarities/aggregator/individual_ratings.pickle 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' &
done
wait

echo '    creating baseline spaces'
python -m code.mds.similarity_spaces.create_baseline_spaces data/Shapes/mds/similarities/aggregator/individual_ratings.pickle data/Shapes/mds/analysis/baseline_vectors.pickle 100 $dimension_limit -n -u -m mean data/Shapes/mds/similarities/aggregator/mean/vectors.pickle median data/Shapes/mds/similarities/aggregator/median/vectors.pickle -s 42


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

# create average category images for best pixel result
python -m code.mds.data_analysis.average_images data/Shapes/mds/similarities/aggregator/individual_ratings.pickle data/Shapes/images/ -o 'data/Shapes/mds/visualizations/average_images/mean/' -r 12 -a min &> 'data/Shapes/mds/visualizations/average_images/mean.txt'
python -m code.mds.data_analysis.average_images data/Shapes/mds/similarities/aggregator/individual_ratings.pickle data/Shapes/images/ -o 'data/Shapes/mds/visualizations/average_images/median/' -r 12 -a min &> 'data/Shapes/mds/visualizations/average_images/median.txt'



echo '    ANN baseline'
for aggregator in $aggregators
do
	echo '        '"$aggregator"
	# if precomputed distances exist: use them; if not: re-compute them
	[ -f 'data/Shapes/mds/analysis/correlations/ann_distances.pickle' ] && image_flag='' || image_flag='-i data/Shapes/images/'
	python -m code.mds.correlations.ann_correlations /tmp/inception 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/analysis/correlations/ann_distances.pickle' 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/ann.csv' $image_flag --kendall -s 42 &> 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/ann_log.txt'

done


echo '    feature baseline'
for aggregator in $aggregators
do
	echo '        '"$aggregator"
	# if precomputed distances exist: use them; if not: re-compute them
	[ -f 'data/Shapes/mds/analysis/correlations/feature_distances.pickle' ] && features_flag='' || features_flag='-f data/Shapes/mds/features/'
	python -m code.mds.correlations.feature_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/analysis/correlations/feature_distances.pickle' 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/features.csv' $features_flag --kendall -s 42 &> 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/features_log.txt' 
 
done

echo '    similarity spaces'
for source_aggregator in $aggregators
do
	# if precomputed distances exist: use them; if not: re-compute them
	[ -f 'data/Shapes/mds/analysis/correlations/mds_from_'"$source_aggregator"'_distances.pickle' ] && vectors_flag='' || vectors_flag='-v data/Shapes/mds/similarities/aggregator/'"$source_aggregator"'/vectors.pickle -b data/Shapes/mds/analysis/baseline_vectors.pickle'
	for target_aggregator in $aggregators
	do
		python -m code.mds.correlations.mds_correlations 'data/Shapes/mds/similarities/aggregator/'"$target_aggregator"'/aggregated_ratings.pickle' 'data/Shapes/mds/analysis/correlations/mds_from_'"$source_aggregator"'_distances.pickle' 'data/Shapes/mds/analysis/correlations/'"$target_aggregator"'/mds_from_'"$source_aggregator"'.csv' $vectors_flag --n_max $dimension_limit --kendall -s 42 &> 'data/Shapes/mds/analysis/correlations/'"$target_aggregator"'/mds_from_'"$source_aggregator"'_log.txt'

	done
done


echo '    creating Shepard plots'
# prepare setup for MEAN
# best pixel (fixed)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_fixed.png -p min -b 24 -d Euclidean' > data/Shapes/mds/analysis/correlations/mean/shepard.config
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_fixed_optimized.png -p min -b 24 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best pixel (optimized)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_optimized.png -p min -b 2 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best ANN (fixed)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_ann_fixed.png -a -d Manhattan' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best ANN (optimized)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_ann_optimized.png -a -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_preattentive_fixed.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_preattentive_optimized.png -f FORM-LINES-ORIENTATION-artificial -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_attentive_fixed.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_attentive_optimized.png -f FORM-LINES-ORIENTATION-artificial -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config

# prepare setup for MEDIAN
# best pixel (fixed)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_fixed.png -p min -b 24 -d Euclidean' > data/Shapes/mds/analysis/correlations/median/shepard.config
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_fixed_optimized.png -p min -b 24 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best pixel (optimized)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_optimized.png -p min -b 26 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best ANN (fixed)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_ann_fixed.png -a -d Manhattan' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best ANN (optimized)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_ann_optimized.png -a -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_preattentive_fixed.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_preattentive_optimized.png -f FORM-LINES-ORIENTATION-artificial -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_attentive_fixed.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_attentive_optimized.png -f FORM-LINES-ORIENTATION-artificial -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config

# add all MDS spaces
for aggregator in $aggregators
do
	for i in  `seq 1 $visualization_limit`
	do
		echo 'data/Shapes/mds/analysis/correlations/mds_from_'"$aggregator"'_distances.pickle data/Shapes/mds/visualizations/correlations/'"$aggregator"'/shepard/mds_'"$i"'.png -m '"$i"' -d Euclidean' >> 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/shepard.config'
	done
done

# now execute all setups
for aggregator in $aggregators
do
	while IFS= read -r line
	do
		python -m code.mds.correlations.shepard_diagram 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/aggregated_ratings.pickle' $line &
	done < 'data/Shapes/mds/analysis/correlations/'"$aggregator"'/shepard.config'
done
wait

# analyzing conceptual regions
# ----------------------------

echo 'analyzing conceptual regions'
echo '    overlap of convex hulls'
for aggregator in $aggregators
do
	for i in `seq 1 $convexity_limit`
	do
		python -m code.mds.regions.analyze_overlap 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' $i 'data/Shapes/mds/analysis/regions/'"$aggregator"'/overlap.csv' -b data/Shapes/mds/analysis/baseline_vectors.pickle &
	done
done
wait

echo '    size'
for aggregator in $aggregators
do
	for i in `seq 1 $dimension_limit`
	do	
		python -m code.mds.regions.analyze_concept_size 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' $i 'data/Shapes/mds/analysis/regions/'"$aggregator"'/size.csv' -b data/Shapes/mds/analysis/baseline_vectors.pickle &
	done
done
wait



# analyzing interpretable directions
# ----------------------------------
echo 'analyzing interpretable directions'

echo '    finding directions'
for aggregator in $aggregators
do
	for direction in $directions
	do
		for i in `seq 1 $dimension_limit`
		do
			python -m code.mds.directions.find_directions 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' $i 'data/Shapes/mds/features/'"$direction"'.pickle' 'data/Shapes/mds/data_set/spaces/directions/'"$aggregator"'/'"$direction"'.csv' -b data/Shapes/mds/analysis/baseline_vectors.pickle &
		done
		wait
	done
done

echo '    comparing directions'
for aggregator in $aggregators
do
	python -m code.mds.directions.compare_directions 'data/Shapes/mds/data_set/spaces/directions/'"$aggregator"'/' $dimension_limit 'data/Shapes/mds/analysis/directions/'"$aggregator"'/similarities.csv' &
done
wait

echo '    aggregating results for analysis'
for aggregator in $aggregators
do
	python -m code.mds.directions.aggregate_direction_results 'data/Shapes/mds/data_set/spaces/directions/'"$aggregator"'/' $dimension_limit 'data/Shapes/mds/analysis/directions/'"$aggregator"'/aggregated/' &

done
wait

echo '    filtering directions for visualization'
for aggregator in $aggregators
do
	for direction in $directions
	do
		python -m code.mds.directions.filter_directions 'data/Shapes/mds/data_set/spaces/directions/'"$aggregator"'/'"$direction"'.csv' $direction $dimension_limit 'data/Shapes/mds/analysis/directions/'"$aggregator"'/filtered.csv' -k 0.8 -s 0.7 &
	done
done
wait



# visualizing spaces
# ------------------

echo 'visualizing MDS spaces'
for aggregator in $aggregators
do
	# clean
	python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/clean/' -i data/Shapes/images/ -m $visualization_limit &
	
	# only regions
	python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/regions/' -i data/Shapes/images/ -m $visualization_limit -r &
	
	for criterion in $criteria
	do
		# only directions
		python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/directions/'"$criterion"'/' -i data/Shapes/images/ -m $visualization_limit -d 'data/Shapes/mds/analysis/directions/'"$aggregator"'/filtered.csv' -c $criterion &

		# regions and directions
		python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/vectors.pickle' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/full/'"$criterion"'/' -i data/Shapes/images/ -m $visualization_limit -d 'data/Shapes/mds/analysis/directions/'"$aggregator"'/filtered.csv' -c $criterion -r &
	done
done
wait

