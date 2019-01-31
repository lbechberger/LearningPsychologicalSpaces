#!/bin/bash

# use 10 dims as maximum
dims=10

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/similarities data/NOUN/vectors/nonmetric data/NOUN/vectors/metric
mkdir -p data/NOUN/visualizations/spaces/nonmetric data/NOUN/visualizations/spaces/metric
mkdir -p data/NOUN/visualizations/correlations/pixels data/NOUN/visualizations/correlations/nonmetric data/NOUN/visualizations/correlations/metric

# preprocessing
echo 'preprocessing data'
python code/preprocessing/preprocess_NOUN.py data/NOUN/raw_data/NOUN_distance_matrix.csv data/NOUN/similarities/sim.pickle -p > data/NOUN/similarities/log.txt

# run MDS
echo 'running MDS'
echo '    nonmetric'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/nonmetric/ -n 64 -i 1000 -p -d $dims -s 42 > data/NOUN/vectors/nonmetric.csv
echo '    metric'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/metric/ -n 64 -i 1000 -p -m -d $dims -s 42 > data/NOUN/vectors/metric.csv

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    nonmetric'
python code/mds/visualize.py data/NOUN/vectors/nonmetric/ data/NOUN/visualizations/spaces/nonmetric -i data/NOUN/images/
echo '    metric'
python code/mds/visualize.py data/NOUN/vectors/metric/ data/NOUN/visualizations/spaces/metric -i data/NOUN/images/


# run image correlation
echo 'image correlation'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/ -s 300 

# run MDS correlations
echo 'MDS correlation'
echo '    nonmetric'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/nonmetric/ -o data/NOUN/visualizations/correlations/nonmetric/ --n_max $dims
echo '    metric'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/metric/ -o data/NOUN/visualizations/correlations/metric/ --n_max $dims

# visualize correlation results
echo 'visualizing correlation'
echo '    nonmetric'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/nonmetric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/nonmetric/sim-MDS.csv > data/NOUN/visualizations/correlations/nonmetric/best.txt
echo '    metric'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/metric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/metric/best.txt


