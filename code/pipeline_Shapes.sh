#!/bin/bash

# use 10 dims as maximum
dims=10

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/similarities 
mkdir -p data/Shapes/vectors/python/nonmetric data/Shapes/vectors/python/metric
mkdir -p data/Shapes/vectors/R/nonmetric data/Shapes/vectors/R/metric
mkdir -p data/Shapes/visualizations/spaces/python/nonmetric data/Shapes/visualizations/spaces/python/metric
mkdir -p data/Shapes/visualizations/spaces/R/nonmetric data/Shapes/visualizations/spaces/R/metric
mkdir -p data/Shapes/visualizations/correlations/pixels 
mkdir -p data/Shapes/visualizations/correlations/python/nonmetric data/Shapes/visualizations/correlations/python/metric
mkdir -p data/Shapes/visualizations/correlations/R/nonmetric data/Shapes/visualizations/correlations/R/metric
mkdir -p data/Shapes/visualizations/average_images/283 data/Shapes/visualizations/average_images/100 data/Shapes/visualizations/average_images/50 
mkdir -p data/Shapes/visualizations/average_images/20 data/Shapes/visualizations/average_images/10 data/Shapes/visualizations/average_images/5


# preprocessing
echo 'preprocessing data'
echo '    read CSV files'
python code/preprocessing/preprocess_Shapes.py data/Shapes/raw_data/within.csv data/Shapes/raw_data/within_between.csv data/Shapes/raw_data/data.pickle
echo '    computing similarities'
python code/preprocessing/compute_similarities.py data/Shapes/raw_data/data.pickle data/Shapes/similarities/sim.pickle -s between -l -p > data/Shapes/similarities/log.txt
echo '    analyzing similarities'
python code/preprocessing/analyze_similarities.py data/Shapes/raw_data/data.pickle -s between -o data/Shapes/similarities/ > data/Shapes/similarities/analysis.txt
echo '    creating average images'
echo '        283'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/283/ -r 283
echo '        100'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/100/ -r 100
echo '        50'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/50/ -r 50
echo '        20'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/20/ -r 20
echo '        10'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/10/ -r 10
echo '        5'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/5/ -r 5
echo '    creating CSV files'
python code/preprocessing/pickle_to_csv.py data/Shapes/similarities/sim.pickle data/Shapes/similarities/


# run MDS
echo 'running MDS'
echo '    nonmetric SMACOF'
python code/mds/mds.py data/Shapes/similarities/sim.pickle -e data/Shapes/vectors/python/nonmetric/ -n 64 -i 1000 -p -d $dims -s 42 > data/Shapes/vectors/python/nonmetric.csv
echo '    metric SMACOF'
python code/mds/mds.py data/Shapes/similarities/sim.pickle -e data/Shapes/vectors/python/metric/ -n 64 -i 1000 -p -m -d $dims -s 42 > data/Shapes/vectors/python/metric.csv
echo '    metric Eigenvalue'
Rscript code/mds/mds.r -d data/Shapes/similarities/distance_matrix.csv -i data/Shapes/similarities/item_names.csv -o data/Shapes/vectors/R/metric/ -p -k $dims --metric > data/Shapes/vectors/R/metric.csv
echo '    nonmetric Kruskal'
Rscript code/mds/mds.r -d data/Shapes/similarities/distance_matrix.csv -i data/Shapes/similarities/item_names.csv -o data/Shapes/vectors/R/nonmetric/ -n 64 -m 1000 -p -k $dims -s 42 > data/Shapes/vectors/R/nonmetric.csv

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    nonmetric SMACOF'
python code/mds/visualize.py data/Shapes/vectors/python/nonmetric/ data/Shapes/visualizations/spaces/python/nonmetric -i data/Shapes/images/
echo '    metric SMACOF'
python code/mds/visualize.py data/Shapes/vectors/python/metric/ data/Shapes/visualizations/spaces/python/metric -i data/Shapes/images/
echo '    metric Eigenvalue'
python code/mds/visualize.py data/Shapes/vectors/R/metric/ data/Shapes/visualizations/spaces/R/metric -i data/Shapes/images/
echo '    nonmetric Kruskal'
python code/mds/visualize.py data/Shapes/vectors/R/nonmetric/ data/Shapes/visualizations/spaces/R/nonmetric -i data/Shapes/images/


# TODO analyze convexity
# TODO analyze interpretable directions


# run image correlation
echo 'image correlation'
python code/correlations/image_correlations.py data/Shapes/similarities/sim.pickle data/Shapes/images/ -o data/Shapes/visualizations/correlations/pixels/ -s 283 

# run MDS correlations
echo 'MDS correlation'
echo '    nonmetric SMACOF'
python code/correlations/mds_correlations.py data/Shapes/similarities/sim.pickle data/Shapes/vectors/python/nonmetric/ -o data/Shapes/visualizations/correlations/python/nonmetric/ --n_max $dims
echo '    metric SMACOF'
python code/correlations/mds_correlations.py data/Shapes/similarities/sim.pickle data/Shapes/vectors/python/metric/ -o data/Shapes/visualizations/correlations/python/metric/ --n_max $dims
echo '    metric Eigenvalue'
python code/correlations/mds_correlations.py data/Shapes/similarities/sim.pickle data/Shapes/vectors/R/metric/ -o data/Shapes/visualizations/correlations/R/metric/ --n_max $dims
echo '    nonmetric Kruskal'
python code/correlations/mds_correlations.py data/Shapes/similarities/sim.pickle data/Shapes/vectors/R/nonmetric/ -o data/Shapes/visualizations/correlations/R/nonmetric/ --n_max $dims

# visualize correlation results
echo 'visualizing correlation'
echo '    nonmetric SMACOF'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/python/nonmetric/ data/Shapes/visualizations/correlations/pixels/sim.csv data/Shapes/visualizations/correlations/python/nonmetric/sim-MDS.csv > data/Shapes/visualizations/correlations/python/nonmetric/best.txt
echo '    metric SMACOF'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/python/metric/ data/Shapes/visualizations/correlations/pixels/sim.csv data/Shapes/visualizations/correlations/python/metric/sim-MDS.csv > data/NOUN/visualizations/correlations/python/metric/best.txt
echo '    metric Eigenvalue'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/R/metric/ data/Shapes/visualizations/correlations/pixels/sim.csv data/Shapes/visualizations/correlations/R/metric/sim-MDS.csv > data/Shapes/visualizations/correlations/R/metric/best.txt
echo '    nonmetric Kruskal'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/R/nonmetric/ data/Shapes/visualizations/correlations/pixels/sim.csv data/Shapes/visualizations/correlations/R/nonmetric/sim-MDS.csv > data/Shapes/visualizations/correlations/R/nonmetric/best.txt
