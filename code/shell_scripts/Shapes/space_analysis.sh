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
	mkdir -p 'data/Shapes/mds/vectors/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/clean/'
	for criterion in $criteria
	do
		mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'"$criterion"'/'
	done
	mkdir -p 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/scatter/'
	mkdir -p 'data/Shapes/mds/visualizations/average_images/'"$aggregator"'/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/'
	mkdir -p 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/aggregated/'
done



# RQ5: How good are the baselines (pixel, ANN, features)?
# -------------------------------------------------------

echo 'RQ5: How good are the baselines (pixel, ANN, features)?'

for aggregator in $aggregators
do
	echo '    looking at '"$aggregator"' matrix'

	# run pixel baseline
	python -m code.mds.correlations.pixel_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel.csv' -w 283 -g --spearman -s 42 &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel-log.txt' 

	# run ANN baseline
	python -m code.mds.correlations.ann_correlations /tmp/inception 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/ann.csv' --spearman -s 42 &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/ann-log.txt'

	# run feature baseline
	python -m code.mds.correlations.feature_correlations 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/regression/' -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/features.csv' --spearman -s 42 &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/features-log.txt' 
done


echo '   creating scatter plots of best fits'

# prepare setup for MEAN
# best pixel (fixed)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_pixel_fixed.png -p min -i data/Shapes/images/ -b 24 -d Euclidean -g' > data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_pixel_fixed_optimized.png -p min -i data/Shapes/images/ -b 24 -d Euclidean -g -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best pixel (optimized)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_pixel_optimized.png -p min -i data/Shapes/images/ -b 2 -d Euclidean -g -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best ANN (fixed)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_ann_fixed.png -a /tmp/inception -i data/Shapes/images/ -d Manhattan' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best ANN (optimized)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_ann_optimized.png -a /tmp/inception -i data/Shapes/images/ -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_feature_preattentive_fixed.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_feature_preattentive_optimized.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION-artificial -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_feature_attentive_fixed.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/visualizations/correlations/mean/scatter/best_feature_attentive_optimized.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION-artificial -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/mean/correlations/scatter.config

# prepare setup for MEDIAN
# best pixel (fixed)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_pixel_fixed.png -p min -i data/Shapes/images/ -b 24 -d Euclidean -g' > data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_pixel_fixed_optimized.png -p min -i data/Shapes/images/ -b 24 -d Euclidean -g -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best pixel (optimized)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_pixel_optimized.png -p min -i data/Shapes/images/ -b 26 -d Euclidean -g -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best ANN (fixed)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_ann_fixed.png -a /tmp/inception -i data/Shapes/images/ -d Manhattan' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best ANN (optimized)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_ann_optimized.png -a /tmp/inception -i data/Shapes/images/ -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_feature_preattentive_fixed.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_feature_preattentive_optimized.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION-artificial -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_feature_attentive_fixed.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/visualizations/correlations/median/scatter/best_feature_attentive_optimized.png -f data/Shapes/mds/regression/ --space FORM-LINES-ORIENTATION-artificial -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/aggregator/median/correlations/scatter.config

# now execute all setups
for aggregator in $aggregators
do
	while IFS= read -r line
	do
		python -m code.mds.correlations.scatter_plot 'data/Shapes/mds/similarities/aggregator/'"$aggregator"'/sim.pickle' $line
	done < 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/scatter.config'
done


# create average category images for best pixel result (MEAN)
python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/mean/' -r 12 -a min &> 'data/Shapes/mds/visualizations/average_images/mean.txt'

# create average category images for best pixel result (MEDIAN)
python -m code.mds.preprocessing.average_images data/Shapes/raw_data/preprocessed/data_visual.pickle data/Shapes/images/ -s between -o 'data/Shapes/mds/visualizations/average_images/median/' -r 12 -a min &> 'data/Shapes/mds/visualizations/average_images/median.txt'



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
		python -m code.mds.correlations.mds_correlations 'data/Shapes/mds/similarities/aggregator/'"$target_aggregator"'/sim.pickle' 'data/Shapes/mds/vectors/'"$source_aggregator"'/' -o 'data/Shapes/mds/analysis/aggregator/'"$source_aggregator"'/correlations/mds_to_'"$target_aggregator"'.csv' --n_max $dimension_limit --spearman -s 42 &> 'data/Shapes/mds/analysis/aggregator/'"$source_aggregator"'/correlations/mds_to_'"$target_aggregator"'-log.txt'&
	done
done
wait

# RQ7: How well-shaped are the conceptual regions?
# ------------------------------------------------

echo 'RQ7: How well-shaped are the conceptual regions?'

for aggregator in $aggregators
do
	for i in `seq 1 $convexity_limit`
	do
		# check whether regions are convex
		python -m code.mds.similarity_spaces.analyze_convexity 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' data/Shapes/raw_data/preprocessed/data_visual.pickle $i -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/convexities.csv' -b -r 100 -s 42 > 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'"$i"'D-convexity.txt' &
		# check how large the regions are
		python -m code.mds.similarity_spaces.analyze_concept_size 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv'  data/Shapes/raw_data/preprocessed/data_visual.pickle $i -o 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/densities.csv' -b -r 100 -s 42 > 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/convexity/'"$i"'D-density.txt' &
	done
done
wait

# RQ8: Are the (psychological) features reflected as interpretable directions in the similarity spaces?
# -----------------------------------------------------------------------------------------------------

echo 'RQ8: Are the (psychological) features reflected as interpretable directions in the similarity spaces?'

echo '    finding directions'
for aggregator in $aggregators
do
	for direction in $directions
	do
		for i in `seq 1 $dimension_limit`
		do
			python -m code.mds.similarity_spaces.find_directions 'data/Shapes/mds/vectors/'"$aggregator"'/'"$i"'D-vectors.csv' $i 'data/Shapes/mds/classification/'"$direction"'.pickle' 'data/Shapes/mds/regression/'"$direction"'.pickle' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/'"$direction"'.csv' &
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

echo '    filtering and aggregating directions'
for aggregator in $aggregators
do
	for direction in $directions
	do
		python -m code.mds.similarity_spaces.aggregate_direction_results 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/' $dimension_limit 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/aggregated/'

		python -m code.mds.similarity_spaces.filter_directions 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/raw/'"$direction"'.csv' $direction $dimension_limit 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/filtered.csv' -k 0.8 -s 0.7 &
	done
done
wait


# Some additional visualizations
# ------------------------------

echo 'Some additional visualizations'

# visualize correlation results
echo '    visualizing correlations'
for aggregator in $aggregators
do
	python -m code.mds.correlations.visualize_correlations -o 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/pixel.csv' 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/mds_to_'"$aggregator"'.csv' --spearman &> 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/correlations/best.txt' &
done
wait

# visualize MDS spaces
echo '    visualizing MDS spaces'
for aggregator in $aggregators
do
	# without directions
	python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/clean/' -i data/Shapes/images/ -m $visualization_limit &
	
	# for each evaluation criterion also with the corresponding directions
	for criterion in $criteria
	do
		python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'"$criterion"'/' -i data/Shapes/images/ -m $visualization_limit -d 'data/Shapes/mds/analysis/aggregator/'"$aggregator"'/directions/filtered.csv' -c $criterion &
	done
done
wait
