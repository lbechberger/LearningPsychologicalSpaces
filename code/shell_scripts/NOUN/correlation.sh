#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/analysis/pixel_correlations/rgb data/NOUN/analysis/pixel_correlations/grey data/NOUN/analysis/inception
mkdir -p data/NOUN/analysis/classical/ data/NOUN/analysis/Kruskal/ data/NOUN/analysis/metric_SMACOF/ data/NOUN/analysis/nonmetric_SMACOF/ data/NOUN/analysis/HorstHout/
mkdir -p data/NOUN/visualizations/correlations/rgb data/NOUN/visualizations/correlations/grey 

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
