#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/analysis/pixel_correlations/ data/NOUN/analysis/ANN_correlations
mkdir -p data/NOUN/analysis/classical/ data/NOUN/analysis/Kruskal/ data/NOUN/analysis/metric_SMACOF/ data/NOUN/analysis/nonmetric_SMACOF/ data/NOUN/analysis/HorstHout/
mkdir -p data/NOUN/visualizations/correlations

# run pixel-based correlation
echo 'pixel-based correlation'
python -m code.correlations.image_correlations data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/analysis/pixel_correlations/ -s 300

# run MDS correlations
echo 'MDS correlation'
echo '    classical'
python -m code.correlations.mds_correlations data/NOUN/similarities/sim.pickle data/NOUN/vectors/classical/ -o data/NOUN/analysis/classical/ --n_max $dims &
echo '    Kruskal'
python -m code.correlations.mds_correlations data/NOUN/similarities/sim.pickle data/NOUN/vectors/Kruskal/ -o data/NOUN/analysis/Kruskal/ --n_max $dims &
echo '    nonmetric SMACOF'
python -m code.correlations.mds_correlations data/NOUN/similarities/sim.pickle data/NOUN/vectors/nonmetric_SMACOF/ -o data/NOUN/analysis/nonmetric_SMACOF/ --n_max $dims &
echo '    metric SMACOF' 
python -m code.correlations.mds_correlations data/NOUN/similarities/sim.pickle data/NOUN/vectors/metric_SMACOF/ -o data/NOUN/analysis/metric_SMACOF/ --n_max $dims &
echo '    Horst and Hout 4D'
python -m code.correlations.mds_correlations data/NOUN/similarities/sim.pickle data/NOUN/vectors/HorstHout/ -o data/NOUN/analysis/HorstHout/ --n_min 4 --n_max 4 &
wait

# run ANN-based correlation
echo 'ANN-based correlation'
python -m code.correlations.inception_correlations /tmp/inception data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/analysis/ANN_correlations/ 


# visualize correlation results
echo 'visualizing correlation results'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/ data/NOUN/analysis/pixel_correlations/sim.csv data/NOUN/analysis/classical/sim-MDS.csv &> data/NOUN/visualizations/correlations/best.txt &
wait
