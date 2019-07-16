#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims="${dims:-10}"
max="${max:-5}"

default_spaces=("classical Kruskal metric_SMACOF nonmetric_SMACOF")
spaces="${algorithms:-$default_spaces}"

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/mds/correlations/ data/NOUN/mds/visualizations/correlations/

# run pixel-based correlation
echo 'pixel-based correlation'
python -m code.mds.correlations.pixel_correlations data/NOUN/mds/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/mds/correlations/pixel.csv -s 300

# run ANN-based correlation
echo 'ANN-based correlation'
python -m code.mds.correlations.ann_correlations /tmp/inception data/NOUN/mds/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/mds/correlations/ann.csv 

# run MDS correlations
echo 'MDS correlation'
for space in $spaces
do
	echo '    '"$space"
	python -m code.mds.correlations.mds_correlations data/NOUN/mds/similarities/sim.pickle 'data/NOUN/mds/vectors/'"$space"'/' -o 'data/NOUN/mds/correlations/'"$space"'.csv' --n_max $dims &
done
python -m code.mds.correlations.mds_correlations data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/vectors/HorstHout/ -o data/NOUN/mds/correlations/HorstHout.csv --n_min 4 --n_max 4 &
wait

# visualize correlation results
echo 'visualizing correlation results'
# overview graphs
python -m code.mds.correlations.visualize_correlations -o data/NOUN/mds/visualizations/correlations/ data/NOUN/mds/correlations/pixel.csv data/NOUN/mds/correlations/classical.csv &> data/NOUN/mds/correlations/best.txt &
# scatter plot for 2D MDS
python -m code.mds.correlations.scatter_plot data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/visualizations/correlations/scatter_MDS.png --mds data/NOUN/mds/vectors/nonmetric_SMACOF/2D-vectors.csv -d Euclidean &
# scatter plot for ANN baseline
python -m code.mds.correlations.scatter_plot data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/visualizations/correlations/scatter_ANN.png --ann /tmp/inception -d Manhattan -i data/NOUN/images/ &
# scatter plot for best pixel baseline
python -m code.mds.correlations.scatter_plot data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/visualizations/correlations/scatter_pixel.png --pixel min -b 19 -d Euclidean -i data/NOUN/images/ &
wait


