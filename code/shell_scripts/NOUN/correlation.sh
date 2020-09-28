#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims="${dims:-10}"
max="${max:-5}"

default_correlation_metrics=("--pearson --spearman --kendall --r2_linear --r2_isotonic")
correlation_metrics="${correlation_metrics:-$default_correlation_metrics}"

default_algorithms=("classical Kruskal metric_SMACOF nonmetric_SMACOF")
spaces="${algorithms:-$default_algorithms}"

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/mds/correlations/pixel_distances/ data/NOUN/mds/visualizations/correlations/scatter

# run pixel-based correlation
echo 'pixel-based correlation'
[ -f 'data/NOUN/mds/correlations/pixel_distances/283-max-Euclidean.pickle' ] && image_flag='' || image_flag='-i data/NOUN/images/'
python -m code.mds.correlations.pixel_correlations data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/pixel_distances/ data/NOUN/mds/correlations/pixel.csv $image_flag -w 300 $correlation_metrics -s 42 &> data/NOUN/mds/correlations/pixel-log.txt

# run ANN-based correlation
echo 'ANN-based correlation'
[ -f 'data/NOUN/mds/correlations/ann_distances.pickle' ] && image_flag='' || image_flag='-i data/NOUN/images/'
python -m code.mds.correlations.ann_correlations /tmp/inception data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/ann-distances.pickle data/NOUN/mds/correlations/ann.csv $image_flag $correlation_metrics -s 42 &> data/NOUN/mds/correlations/ann-log.txt

# run MDS correlations along with baselines
echo 'MDS correlation'
python -m code.mds.similarity_spaces.create_baseline_spaces data/NOUN/mds/raw_data/data.pickle data/NOUN/mds/correlations/baseline_vectors.pickle 100 $dims -n -u -s 42
for space in $spaces
do
	echo '    '"$space"
	# if precomputed distances exist: use them; if not: re-compute them
	[ -f 'data/NOUN/mds/correlations/'"$space"'_distances.pickle' ] && vectors_flag='' || vectors_flag='-v data/NOUN/mds/vectors/'"$space"'/vectors.pickle'
	[ "$space" == "nonmetric_SMACOF" ] && baseline_flag='-b data/NOUN/mds/correlations/baseline_vectors.pickle' || baseline_flag=''
	python -m code.mds.correlations.mds_correlations data/NOUN/mds/similarities/sim.pickle 'data/NOUN/mds/correlations/'"$space"'_distances.pickle' 'data/NOUN/mds/correlations/'"$space"'.csv' --n_max $dims $correlation_metrics -s 42 $vectors_flag $baseline_flag &> 'data/NOUN/mds/correlations/'"$space"'-log.txt' &
done
python -m code.mds.correlations.mds_correlations data/NOUN/mds/similarities/sim.pickle 'data/NOUN/mds/correlations/HorstHout_distances.pickle' data/NOUN/mds/correlations/HorstHout.csv --n_min 4 --n_max 4 -v 'data/NOUN/mds/vectors/HorstHout/vectors.pickle' $correlation_metrics -s 42 &> data/NOUN/mds/correlations/HorstHout-log.txt &
wait

# visualize correlation results
echo 'visualizing correlation results'
# overview graphs
python -m code.mds.correlations.visualize_pixel_correlations data/NOUN/mds/correlations/pixel.csv data/NOUN/mds/visualizations/correlations/ &> data/NOUN/mds/correlations/best_pixel.txt &
# Shepard plots for nonmetric SMACOF
for i in `seq 1 $max`
do
	python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle 'data/NOUN/mds/correlations/nonmetric_SMACOF_distances.pickle' 'data/NOUN/mds/visualizations/correlations/scatter/nonmetric_SMACOF_'"$i"'D.png' --mds $i -d Euclidean &	
done
# Shepard plot for ANN baseline
python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/ann_distances.pickle data/NOUN/mds/visualizations/correlations/scatter/ANN_fixed.png --ann -d Manhattan  &
python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/ann_distances.pickle data/NOUN/mds/visualizations/correlations/scatter/ANN_optimized.png --ann  -d Euclidean -n 5 -s 42 &
# Shepard plot for best pixel baseline
python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/pixel_distances/ data/NOUN/mds/visualizations/correlations/scatter/pixel_fixed.png --pixel min -b 18 -d Manhattan  &
python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/pixel_distances/ data/NOUN/mds/visualizations/correlations/scatter/pixel_fixed_optimized.png --pixel min -b 18 -d Manhattan  -n 5 -s 42&
python -m code.mds.correlations.shepard_diagram data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/correlations/pixel_distances/ data/NOUN/mds/visualizations/correlations/scatter/pixel_optimized.png --pixel max -b 1 -d Euclidean -n 5 -s 42 &
wait


