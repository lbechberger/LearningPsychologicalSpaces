#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/similarities/mean data/Shapes/similarities/median 
mkdir -p data/Shapes/vectors/Kruskal_mean/ data/Shapes/vectors/SMACOF_median/ data/Shapes/vectors/SMACOF_mean/
mkdir -p data/Shapes/visualizations/spaces/Kruskal_mean data/Shapes/visualizations/spaces/SMACOF_mean data/Shapes/visualizations/spaces/SMACOF_median
mkdir -p data/Shapes/visualizations/correlations/pixels/mean data/Shapes/visualizations/correlations/pixels/median
mkdir -p data/Shapes/visualizations/correlations/Kruskal_mean data/Shapes/visualizations/correlations/SMACOF_mean data/Shapes/visualizations/correlations/SMACOF_median
mkdir -p data/Shapes/visualizations/average_images/283 data/Shapes/visualizations/average_images/100 data/Shapes/visualizations/average_images/50 
mkdir -p data/Shapes/visualizations/average_images/20 data/Shapes/visualizations/average_images/10 data/Shapes/visualizations/average_images/5
mkdir -p data/Shapes/analysis/Kruskal_mean data/Shapes/analysis/SMACOF_mean data/Shapes/analysis/SMACOF_median


# preprocessing
echo 'preprocessing data'
echo '    reading CSV files'
python code/preprocessing/preprocess_Shapes.py data/Shapes/raw_data/within.csv data/Shapes/raw_data/within_between.csv data/Shapes/raw_data/data.pickle &> data/Shapes/raw_data/preprocess_log.txt
echo '    computing similarities'
echo '        mean'
python code/preprocessing/compute_similarities.py data/Shapes/raw_data/data.pickle data/Shapes/similarities/mean/sim.pickle -s between -l -p &> data/Shapes/similarities/log_mean.txt
echo '        median'
python code/preprocessing/compute_similarities.py data/Shapes/raw_data/data.pickle data/Shapes/similarities/median/sim.pickle -s between -l -p -m &> data/Shapes/similarities/log_median.txt
echo '    analyzing similarities'
python code/preprocessing/analyze_similarities.py data/Shapes/raw_data/data.pickle -s between -o data/Shapes/similarities/ &> data/Shapes/similarities/analysis.txt
echo '    creating average images'
echo '        283'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/283/ -r 283 &> data/Shapes/visualizations/average_images/283.txt
echo '        100'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/100/ -r 100 &> data/Shapes/visualizations/average_images/100.txt
echo '        50'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/50/ -r 50 &> data/Shapes/visualizations/average_images/50.txt
echo '        20'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/20/ -r 20 &> data/Shapes/visualizations/average_images/20.txt
echo '        10'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/10/ -r 10 &> data/Shapes/visualizations/average_images/10.txt
echo '        5'
python code/preprocessing/average_images.py data/Shapes/raw_data/data.pickle data/Shapes/images/ -s between -o data/Shapes/visualizations/average_images/5/ -r 5 &> data/Shapes/visualizations/average_images/5.txt
echo '    creating CSV files'
echo '        mean'
python code/preprocessing/pickle_to_csv.py data/Shapes/similarities/mean/sim.pickle data/Shapes/similarities/mean/
echo '        median'
python code/preprocessing/pickle_to_csv.py data/Shapes/similarities/median/sim.pickle data/Shapes/similarities/median/

# run MDS
echo 'running MDS'
echo '    Kruskal mean'
Rscript code/mds/mds.r -d data/Shapes/similarities/mean/distance_matrix.csv -i data/Shapes/similarities/mean/item_names.csv -o data/Shapes/vectors/Kruskal_mean/ -n 256 -m 1000 -k $dims -s 42 &> data/Shapes/vectors/Kruskal_mean.csv &
echo '    SMACOF mean'
Rscript code/mds/mds.r -d data/Shapes/similarities/mean/distance_matrix.csv -i data/Shapes/similarities/mean/item_names.csv -o data/Shapes/vectors/SMACOF_mean/ -n 256 -m 1000 -k $dims -s 42 --smacof &> data/Shapes/vectors/SMACOF_mean.csv &
echo '    SMACOF median'
Rscript code/mds/mds.r -d data/Shapes/similarities/median/distance_matrix.csv -i data/Shapes/similarities/median/item_names.csv -o data/Shapes/vectors/SMACOF_median/ -n 256 -m 1000 -k $dims -s 42 --smacof &> data/Shapes/vectors/SMACOF_median.csv &
wait

# normalize MDS spaces
echo 'normalizing MDS spaces'
echo '    Kruskal mean'
python code/mds/normalize_spaces.py data/Shapes/vectors/Kruskal_mean/
echo '    SMACOF mean'
python code/mds/normalize_spaces.py data/Shapes/vectors/SMACOF_mean/
echo '    SMACOF median'
python code/mds/normalize_spaces.py data/Shapes/vectors/SMACOF_median/

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    Kruskal mean'
python code/mds/visualize.py data/Shapes/vectors/Kruskal_mean/ data/Shapes/visualizations/spaces/Kruskal_mean -i data/Shapes/images/ -m $max &
echo '    SMACOF mean'
python code/mds/visualize.py data/Shapes/vectors/SMACOF_mean/ data/Shapes/visualizations/spaces/SMACOF_mean -i data/Shapes/images/ -m $max &
echo '    SMACOF median'
python code/mds/visualize.py data/Shapes/vectors/SMACOF_median/ data/Shapes/visualizations/spaces/SMACOF_median -i data/Shapes/images/ -m $max &
wait


# analyze convexity
echo 'analyzing convexity'

echo '    Kruskal mean'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/Kruskal_mean/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/Kruskal_mean/convexities.csv -b -r 100 > 'data/Shapes/analysis/Kruskal_mean/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF mean'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/SMACOF_mean/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/SMACOF_mean/convexities.csv > 'data/Shapes/analysis/SMACOF_mean/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF median'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/SMACOF_median/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/SMACOF_median/convexities.csv > 'data/Shapes/analysis/SMACOF_median/'"$i"'D-convexity.txt' &
done
wait


# analyze interpretable directions
echo 'analyzing interpretable directions'

echo '    Kruskal mean'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/Kruskal_mean/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/Kruskal_mean/interpretabilities.csv -b -r 100 > 'data/Shapes/analysis/Kruskal_mean/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF mean'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/SMACOF_mean/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/SMACOF_mean/interpretabilities.csv > 'data/Shapes/analysis/SMACOF_mean/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF median'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/SMACOF_median/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/SMACOF_median/interpretabilities.csv > 'data/Shapes/analysis/SMACOF_median/'"$i"'D-interpretability.txt' &
done
wait


# run image correlation
echo 'image correlation'
echo '    mean'
python code/correlations/image_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/images/ -o data/Shapes/visualizations/correlations/pixels/mean/ -s 283 -g &
echo '    median'
python code/correlations/image_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/images/ -o data/Shapes/visualizations/correlations/pixels/median/ -s 283 -g &
wait

# run MDS correlations
echo 'MDS correlation'
echo '    Kruskal mean'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/Kruskal_mean/ -o data/Shapes/visualizations/correlations/Kruskal_mean/ --n_max $dims
echo '    SMACOF mean'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/SMACOF_mean/ -o data/Shapes/visualizations/correlations/SMACOF_mean/ --n_max $dims
echo '    SMACOF median'
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/SMACOF_median/ -o data/Shapes/visualizations/correlations/SMACOF_median/ --n_max $dims

# visualize correlation results
echo 'visualizing correlation'
echo '    Kruskal mean'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/Kruskal_mean/ data/Shapes/visualizations/correlations/pixels/mean/sim-g.csv data/Shapes/visualizations/correlations/Kruskal_mean/sim-MDS.csv > data/Shapes/visualizations/correlations/Kruskal_mean/best.txt
echo '    SMACOF mean'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/SMACOF_mean/ data/Shapes/visualizations/correlations/pixels/mean/sim-g.csv data/Shapes/visualizations/correlations/SMACOF_mean/sim-MDS.csv > data/Shapes/visualizations/correlations/SMACOF_mean/best.txt
echo '    SMACOF median'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/SMACOF_median/ data/Shapes/visualizations/correlations/pixels/median/sim-g.csv data/Shapes/visualizations/correlations/SMACOF_median/sim-MDS.csv > data/Shapes/visualizations/correlations/SMACOF_median/best.txt

