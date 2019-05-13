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
mkdir -p data/NOUN/analysis/pixel_correlations/rgb data/NOUN/analysis/pixel_correlations/grey
mkdir -p data/NOUN/analysis/classical/ data/NOUN/analysis/Kruskal/ data/NOUN/analysis/metric_SMACOF/ data/NOUN/analysis/nonmetric_SMACOF/ data/NOUN/analysis/HorstHout/
mkdir -p data/NOUN/visualizations/correlations/rgb data/NOUN/visualizations/correlations/grey 

mkdir -p data/NOUN/dataset/augmented data/NOUN/analysis/features data/NOUN/analysis/inception

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
python code/mds/normalize_spaces.py data/NOUN/vectors/classical/ &
echo '    Kruskal'
python code/mds/normalize_spaces.py data/NOUN/vectors/Kruskal/ &
echo '    nonmetric SMACOF'
python code/mds/normalize_spaces.py data/NOUN/vectors/nonmetric_SMACOF/ &
echo '    metric SMACOF'
python code/mds/normalize_spaces.py data/NOUN/vectors/metric_SMACOF/ &
wait

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
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/analysis/pixel_correlations/rgb/ -s 300 &
echo '    greyscale'
python code/correlations/image_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/analysis/pixel_correlations/grey/ -s 300 -g &
wait

# run MDS correlations
echo 'MDS correlation'
echo '    classical'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/classical/ -o data/NOUN/analysis/classical/ --n_max $dims &
echo '    Kruskal'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/Kruskal/ -o data/NOUN/analysis/Kruskal/ --n_max $dims &
echo '    nonmetric SMACOF'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/nonmetric_SMACOF/ -o data/NOUN/analysis/nonmetric_SMACOF/ --n_max $dims &
echo '    metric SMACOF' 
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/metric_SMACOF/ -o data/NOUN/analysis/metric_SMACOF/ --n_max $dims &
echo '    Horst and Hout 4D'
python code/correlations/mds_correlations.py data/NOUN/similarities/sim.pickle data/NOUN/vectors/HorstHout/ -o data/NOUN/analysis/HorstHout/ --n_min 4 --n_max 4 &
wait

# run inception correlation
echo 'inception correlation'
python code/correlations/inception_correlations.py /tmp/inception data/NOUN/similarities/sim.pickle data/NOUN/images/ -o data/NOUN/analysis/inception/ 


# visualize correlation results
echo 'visualizing correlation'
echo '    RGB'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/rgb/ data/NOUN/analysis/pixel_correlations/rgb/sim.csv data/NOUN/analysis/classical/sim-MDS.csv &> data/NOUN/visualizations/correlations/rgb/best.txt &
echo '    Greyscale'
python code/correlations/visualize_correlations.py -o data/NOUN/visualizations/correlations/grey/ data/NOUN/analysis/pixel_correlations/grey/sim-g.csv data/NOUN/analysis/classical/sim-MDS.csv &> data/NOUN/visualizations/correlations/grey/best.txt &
wait

# machine learning
echo 'preparing data set for machine learning'

echo '    augmentation'
python code/dataset/data_augmentation.py data/NOUN/images/ data/NOUN/dataset/augmented 1000 -s 42 --flip_prob 0.0 --crop_size 0.05 --scale_min 0.9 --scale_max 1.1 --translation 0.1 --sp_noise_prob 0.01 --rotation_angle 15
echo '    regression targets'
python code/dataset/prepare_targets.py data/NOUN/dataset/targets.csv data/NOUN/dataset/targets.pickle -s 42

echo '    feature extraction'
echo '        inception network'
python code/regression/inception_features.py /tmp/inception data/NOUN/dataset/augmented data/NOUN/dataset/features_inception.pickle
echo '        reduced images'
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_7_g.pickle -b 7 -a min -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_18_g.pickle -b 18 -a min -g
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_12_rgb.pickle -b 12 -a min
python code/regression/reduced_image_features.py data/NOUN/dataset/augmented/ data/NOUN/dataset/features_image_min_18_rgb.pickle -b 18 -a min

echo '    cluster analysis'
echo '        inception network'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_inception.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_inception.txt
echo '        reduced images'
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_7_g.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_7_g.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_18_g.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_18_g.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_12_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_12_rgb.txt
python code/regression/cluster_analysis.py data/NOUN/dataset/features_image_min_18_rgb.pickle -n 100 -s 42 > data/NOUN/analysis/features/features_image_min_18_rgb.txt

echo 'running linear regressions'
echo '    baselines'
# TODO
echo '    inception'
# TODO
echo '    reduced images'
# TODO

