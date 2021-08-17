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


echo '    creating Shepard plots'
# prepare setup for MEAN
# best pixel (fixed)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_fixed.png -p min -b 24 -d Euclidean' > data/Shapes/mds/analysis/correlations/mean/shepard.config
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_fixed_optimized.png -p min -b 24 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best pixel (optimized)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/mean/shepard/best_pixel_optimized.png -p max -b 1 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best ANN (fixed)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_ann_fixed.png -a -d Manhattan' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best ANN (optimized)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_ann_optimized.png -a -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_preattentive_fixed.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_preattentive_optimized.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_attentive_fixed.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/correlations/mean/shepard.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/mean/shepard/best_feature_attentive_optimized.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/mean/shepard.config

# prepare setup for MEDIAN
# best pixel (fixed)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_fixed.png -p min -b 24 -d Euclidean' > data/Shapes/mds/analysis/correlations/median/shepard.config
# best pixel (optimized)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_optimized.png -p min -b 24 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best pixel (full resolution)
echo 'data/Shapes/mds/analysis/correlations/pixel_distances/ data/Shapes/mds/visualizations/correlations/median/shepard/best_pixel_full.png -p max -b 1 -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best ANN (fixed)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_ann_fixed.png -a -d Manhattan' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best ANN (optimized)
echo 'data/Shapes/mds/analysis/correlations/ann_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_ann_optimized.png -a -d Euclidean -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (preattentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_preattentive_fixed.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (preattentive, optimimzed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_preattentive_optimized.png -f FORM-LINES-ORIENTATION -t pre-attentive -d InnerProduct -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (attentive, fixed)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_attentive_fixed.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan' >> data/Shapes/mds/analysis/correlations/median/shepard.config
# best feature (attentive, optimized)
echo 'data/Shapes/mds/analysis/correlations/feature_distances.pickle data/Shapes/mds/visualizations/correlations/median/shepard/best_feature_attentive_optimized.png -f FORM-LINES-ORIENTATION -t attentive -d Manhattan -o -s 42 -n 5' >> data/Shapes/mds/analysis/correlations/median/shepard.config

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
