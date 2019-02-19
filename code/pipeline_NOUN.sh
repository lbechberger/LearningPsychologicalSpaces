#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/similarities 
mkdir -p data/NOUN/vectors/classical/ data/NOUN/vectors/Kruskal/ data/NOUN/vectors/metric_SMACOF/ data/NOUN/vectors/nonmetric_SMACOF/
mkdir -p data/NOUN/vectors/HorstHout/
mkdir -p data/NOUN/visualizations/spaces/classical/ data/NOUN/visualizations/spaces/Kruskal/ data/NOUN/visualizations/spaces/metric_SMACOF/ data/NOUN/visualizations/spaces/nonmetric_SMACOF/
mkdir -p data/NOUN/visualizations/spaces/HorstHout
mkdir -p data/NOUN/visualizations/correlations/pixels/rgb data/NOUN/visualizations/correlations/pixels/grey data/NOUN/visualizations/correlations/grey data/NOUN/visualizations/correlations/HorstHout/
mkdir -p data/NOUN/visualizations/correlations/classical/ data/NOUN/visualizations/correlations/Kruskal/ data/NOUN/visualizations/correlations/metric_SMACOF/ data/NOUN/visualizations/correlations/nonmetric_SMACOF/
cp data/NOUN/raw_data/4D-vectors.csv data/NOUN/vectors/HorstHout/4D-vectors.csv

# preprocessing
echo 'preprocessing data'
echo '    reading CSV file'
python code/preprocessing/preprocess_NOUN.py data/NOUN/raw_data/raw_distances.csv data/NOUN/raw_data/data.pickle
echo '    computing similarities'
python code/preprocessing/compute_similarities.py data/NOUN/raw_data/data.pickle data/NOUN/similarities/sim.pickle -s within -l -p &> data/NOUN/similarities/log.txt
echo '    creating CSV files'
python code/preprocessing/pickle_to_csv.py data/NOUN/similarities/sim.pickle data/NOUN/similarities/


# run MDS
echo 'running MDS'
echo '    classical'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/classical/ -n 256 -m 1000 -k $dims -s 42 --metric &> data/NOUN/vectors/classical.txt &
echo '    Kruskal'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/Kruskal/ -n 256 -m 1000 -k $dims -s 42 &> data/NOUN/vectors/Kruskal.txt &
echo '    nonmetric SMACOF'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/nonmetric_SMACOF/ -n 256 -m 1000 -k $dims -s 42 --smacof &> data/NOUN/vectors/nonmetric_smacof.txt &
echo '    metric SMACOF'
Rscript code/mds/mds.r -d data/NOUN/similarities/distance_matrix.csv -i data/NOUN/similarities/item_names.csv -o data/NOUN/vectors/metric_SMACOF/ -n 256 -m 1000 -k $dims -s 42 --metric --smacof &> data/NOUN/vectors/metric_smacof.txt &
wait

# normalize MDS spaces
echo 'normalizing MDS spaces'
echo '    classical'
python code/mds/normalize_spaces.py data/NOUN/vectors/classical/
echo '    Kruskal'
python code/mds/normalize_spaces.py data/NOUN/vectors/Kruskal/
echo '    nonmetric SMACOF'
python code/mds/normalize_spaces.py data/NOUN/vectors/nonmetric_SMACOF/
echo '    metric SMACOF'
python code/mds/normalize_spaces.py data/NOUN/vectors/metric_SMACOF/

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    classical'
python code/mds/visualize.py data/NOUN/vectors/classical/ data/NOUN/visualizations/spaces/classical -i data/NOUN/images/ -m $max &
echo '    Kruskal'
python code/mds/visualize.py data/NOUN/vectors/Kruskal/ data/NOUN/visualizations/spaces/Kruskal -i data/NOUN/images/ -m $max &
echo '    nonmetric SMACOF'
python code/mds/visualize.py data/NOUN/vectors/nonmetric_SMACOF/ data/NOUN/visualizations/spaces/nonmetric_SMACOF -i data/NOUN/images/ -m $max &
echo '    metric SMACOF'
python code/mds/visualize.py data/NOUN/vectors/metric_SMACOF data/NOUN/visualizations/spaces/metric_SMACOF -i data/NOUN/images/ -m $max &
echo '    Horst and Hout 4D'
python code/mds/visualize.py data/NOUN/vectors/HorstHout/ data/NOUN/visualizations/spaces/HorstHout -i data/NOUN/images/ -m $max &
wait

# run image correlation
echo 'image correlation'
echo '    full RGB'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/rgb/ -s 300 &
echo '    greyscale'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/visualizations/correlations/pixels/grey/ -s 300 -g &
wait

# run MDS correlations
echo 'MDS correlation'
echo '    classical'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/classical/ -o data/NOUN/visualizations/correlations/classical/ --n_max $dims
echo '    Kruskal'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/Kruskal/ -o data/NOUN/visualizations/correlations/Kruskal/ --n_max $dims
echo '    nonmetric SMACOF'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/nonmetric_SMACOF/ -o data/NOUN/visualizations/correlations/nonmetric_SMACOF/ --n_max $dims
echo '    metric SMACOF'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/metric_SMACOF/ -o data/NOUN/visualizations/correlations/metric_SMACOF/ --n_max $dims
echo '    Horst and Hout 4D'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/HorstHout/ -o data/NOUN/visualizations/correlations/HorstHout/ --n_min 4 --n_max 4

# visualize correlation results
echo 'visualizing correlation'
echo '    classical'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/classical/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/classical/sim-MDS.csv &> data/NOUN/visualizations/correlations/classical/best.txt
echo '    Kruskal'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/Kruskal/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/Kruskal/sim-MDS.csv &> data/NOUN/visualizations/correlations/Kruskal/best.txt
echo '    metric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/nonmetric_SMACOF/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/nonmetric_SMACOF/sim-MDS.csv > data/NOUN/visualizations/correlations/nonmetric_SMACOF/best.txt
echo '    nonmetric SMACOF'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/metric_SMACOF/ data/NOUN/visualizations/correlations/pixels/rgb/sim.csv data/NOUN/visualizations/correlations/metric_SMACOF/sim-MDS.csv &> data/NOUN/visualizations/correlations/metric_SMACOF/best.txt
echo '    (grey)'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/grey/ data/NOUN/visualizations/correlations/pixels/grey/sim-g.csv data/NOUN/visualizations/correlations/classical/sim-MDS.csv &> data/NOUN/visualizations/correlations/grey/best.txt


# TODO do machine learning
