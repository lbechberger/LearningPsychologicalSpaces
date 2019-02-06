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
mkdir -p data/NOUN/visualizations/correlations/pixels/rgb data/NOUN/visualizations/correlations/pixels/grey
mkdir -p data/NOUN/visualizations/correlations/python/nonmetric data/NOUN/visualizations/correlations/python/metric
mkdir -p data/NOUN/visualizations/correlations/R/nonmetric data/NOUN/visualizations/correlations/R/metric
mkdir -p data/NOUN/visualizations/correlations/python/grey data/NOUN/visualizations/correlations/HorstHout

# preprocessing
echo 'preprocessing data'
echo '    reading CSV file'
python code/preprocessing/preprocess_NOUN.py data/NOUN/raw_data/raw_distances.csv data/NOUN/raw_data/data.pickle
echo '    computing similarities'
python code/preprocessing/compute_similarities.py data/NOUN/raw_data/data.pickle data/NOUN/similarities/sim.pickle -s within -l -p > data/NOUN/similarities/log.txt
echo '    creating CSV files'
python code/preprocessing/pickle_to_csv.py data/NOUN/similarities/sim.pickle data/NOUN/similarities/


# run MDS
echo 'running MDS'
echo '    nonmetric SMACOF'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/python/nonmetric/ -n 256 -i 1000 -p -d $dims -s 42 > data/NOUN/vectors/python/nonmetric.csv
echo '    metric SMACOF'
python code/mds/mds.py data/NOUN/similarities/sim.pickle -e data/NOUN/vectors/python/metric/ -n 256 -i 1000 -p -m -d $dims -s 42 > data/NOUN/vectors/python/metric.csv
echo '    metric Eigenvalue'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/R/metric/ -p -k $dims --metric > data/NOUN/vectors/R/metric.csv
echo '    nonmetric Kruskal'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/R/nonmetric/ -n 256 -m 1000 -p -k $dims -s 42 > data/NOUN/vectors/R/nonmetric.csv

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
echo '    full RGB'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/rgb/ -s 300 
echo '    greyscale'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/grey/ -s 300 -g

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
echo '    Horst and Hout 4D'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/HorstHout/ -o data/NOUN/visualizations/correlations/HorstHout/ --n_min 4 --n_max 4

# visualize correlation results
echo 'visualizing correlation'
echo '    nonmetric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/python/nonmetric/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/python/nonmetric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/nonmetric/best.txt
echo '    metric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/python/metric/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/python/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/metric/best.txt
echo '    metric Eigenvalue'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/R/metric/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/R/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/R/metric/best.txt
echo '    nonmetric Kruskal'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/R/nonmetric/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/R/nonmetric/sim-MDS.csv > data/NOUN/visualizations/correlations/R/nonmetric/best.txt
echo '    metric SMACOF (grey)'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/python/grey/ data/NOUN/visualizations/correlations/pixels/grey/sim-g.csv data/NOUN/visualizations/correlations/python/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/grey/best.txt


# TODO do machine learning
