#!/bin/bash

# use 10 dims as maximum
dims=10

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/similarities 
mkdir -p data/NOUN/vectors/python/nonmetric data/NOUN/vectors/python/metric
mkdir -p data/NOUN/vectors/R/nonmetric data/NOUN/vectors/R/metric
mkdir -p data/NOUN/visualizations/spaces/python/nonmetric data/NOUN/visualizations/spaces/python/metric
mkdir -p data/NOUN/visualizations/spaces/R/nonmetric data/NOUN/visualizations/spaces/R/metric
mkdir -p data/NOUN/visualizations/correlations/pixels 
mkdir -p data/NOUN/visualizations/correlations/python/nonmetric data/NOUN/visualizations/correlations/python/metric
mkdir -p data/NOUN/visualizations/correlations/R/nonmetric data/NOUN/visualizations/correlations/R/metric

# preprocessing
echo 'preprocessing data'
python code/preprocessing/preprocess_NOUN.py data/NOUN/raw_data/NOUN_distance_matrix.csv data/NOUN/similarities/sim.pickle -p > data/NOUN/similarities/log.txt

# run MDS
echo 'running MDS'
echo '    nonmetric SMACOF'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/python/nonmetric/ -n 64 -i 1000 -p -d $dims -s 42 > data/NOUN/vectors/python/nonmetric.csv
echo '    metric SMACOF'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/python/metric/ -n 64 -i 1000 -p -m -d $dims -s 42 > data/NOUN/vectors/python/metric.csv
echo '    metric Eigenvalue'
Rscript code/mds/mds.r -d data/NOUN/raw_data/NOUN_distance_matrix.csv -i data/NOUN/raw_data/item_names.csv -o data/NOUN/vectors/R/metric/ -p -k $dims --metric > data/NOUN/vectors/R/metric.csv
echo '    nonmetric Kruskal'
Rscript code/mds/mds.r -d data/NOUN/raw_data/NOUN_distance_matrix.csv -i data/NOUN/raw_data/item_names.csv -o data/NOUN/vectors/R/nonmetric/ -n 64 -m 1000 -p -k $dims -s 42 > data/NOUN/vectors/R/nonmetric.csv

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    nonmetric SMACOF'
python code/mds/visualize.py data/NOUN/vectors/python/nonmetric/ data/NOUN/visualizations/spaces/python/nonmetric -i data/NOUN/images/
echo '    metric SMACOF'
python code/mds/visualize.py data/NOUN/vectors/python/metric/ data/NOUN/visualizations/spaces/python/metric -i data/NOUN/images/
echo '    metric Eigenvalue'
python code/mds/visualize.py data/NOUN/vectors/R/metric/ data/NOUN/visualizations/spaces/R/metric -i data/NOUN/images/
echo '    nonmetric Kruskal'
python code/mds/visualize.py data/NOUN/vectors/R/nonmetric/ data/NOUN/visualizations/spaces/R/nonmetric -i data/NOUN/images/


# run image correlation
echo 'image correlation'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/ -s 300 

# run MDS correlations
echo 'MDS correlation'
echo '    nonmetric SMACOF'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/python/nonmetric/ -o data/NOUN/visualizations/correlations/python/nonmetric/ --n_max $dims
echo '    metric SMACOF'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/python/metric/ -o data/NOUN/visualizations/correlations/python/metric/ --n_max $dims
echo '    metric Eigenvalue'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/R/metric/ -o data/NOUN/visualizations/correlations/R/metric/ --n_max $dims
echo '    nonmetric Kruskal'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/R/nonmetric/ -o data/NOUN/visualizations/correlations/R/nonmetric/ --n_max $dims

# visualize correlation results
echo 'visualizing correlation'
echo '    nonmetric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/python/nonmetric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/python/nonmetric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/nonmetric/best.txt
echo '    metric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/python/metric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/python/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/metric/best.txt
echo '    metric Eigenvalue'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/R/metric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/R/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/R/metric/best.txt
echo '    nonmetric Kruskal'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/R/nonmetric/ data/NOUN/visualizations/correlations/pixels/sim.csv data/NOUN/visualizations/correlations/R/nonmetric/sim-MDS.csv > data/NOUN/visualizations/correlations/R/nonmetric/best.txt
