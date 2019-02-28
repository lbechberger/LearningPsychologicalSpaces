#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/Shapes/similarities/mean data/Shapes/similarities/median 
mkdir -p data/Shapes/vectors/mean/Kruskal/ data/Shapes/vectors/mean/SMACOF
mkdir -p data/Shapes/vectors/median/primary/ data/Shapes/vectors/median/secondary/ data/Shapes/vectors/median/tertiary/
mkdir -p data/Shapes/visualizations/spaces/mean/Kruskal data/Shapes/visualizations/spaces/mean/SMACOF
mkdir -p data/Shapes/visualizations/spaces/median/primary data/Shapes/visualizations/spaces/median/secondary data/Shapes/visualizations/spaces/median/tertiary
mkdir -p data/Shapes/visualizations/average_images/283 data/Shapes/visualizations/average_images/100 data/Shapes/visualizations/average_images/50 
mkdir -p data/Shapes/visualizations/average_images/20 data/Shapes/visualizations/average_images/10 data/Shapes/visualizations/average_images/5
mkdir -p data/Shapes/analysis/mean/Kruskal/convexity data/Shapes/analysis/mean/Kruskal/interpretability data/Shapes/analysis/mean/Kruskal/correlation_mean data/Shapes/analysis/mean/Kruskal/correlation_median
mkdir -p data/Shapes/analysis/mean/SMACOF/convexity data/Shapes/analysis/mean/SMACOF/interpretability data/Shapes/analysis/mean/SMACOF/correlation_mean data/Shapes/analysis/mean/SMACOF/correlation_median
mkdir -p data/Shapes/analysis/median/primary/convexity data/Shapes/analysis/median/primary/interpretability data/Shapes/analysis/median/primary/correlation_mean data/Shapes/analysis/median/primary/correlation_median
mkdir -p data/Shapes/analysis/median/secondary/convexity data/Shapes/analysis/median/secondary/interpretability data/Shapes/analysis/median/secondary/correlation_mean data/Shapes/analysis/median/secondary/correlation_median
mkdir -p data/Shapes/analysis/median/tertiary/convexity data/Shapes/analysis/median/tertiary/interpretability data/Shapes/analysis/median/tertiary/correlation_mean data/Shapes/analysis/median/tertiary/correlation_median
mkdir -p data/Shapes/analysis/pixel_correlations/mean data/Shapes/analysis/pixel_correlations/median
mkdir -p data/Shapes/visualizations/correlations/mean/Kruskal/ data/Shapes/visualizations/correlations/mean/SMACOF/
mkdir -p data/Shapes/visualizations/correlations/median/primary/ data/Shapes/visualizations/correlations/median/secondary/ data/Shapes/visualizations/correlations/median/tertiary/


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
Rscript code/mds/mds.r -d data/Shapes/similarities/mean/distance_matrix.csv -i data/Shapes/similarities/mean/item_names.csv -o data/Shapes/vectors/mean/Kruskal/ -n 256 -m 1000 -k $dims -s 42 &> data/Shapes/vectors/mean/Kruskal.csv &
echo '    SMACOF mean'
Rscript code/mds/mds.r -d data/Shapes/similarities/mean/distance_matrix.csv -i data/Shapes/similarities/mean/item_names.csv -o data/Shapes/vectors/mean/SMACOF/ -n 256 -m 1000 -k $dims -s 42 --smacof &> data/Shapes/vectors/mean/SMACOF.csv &
echo '    SMACOF median primary'
Rscript code/mds/mds.r -d data/Shapes/similarities/median/distance_matrix.csv -i data/Shapes/similarities/median/item_names.csv -o data/Shapes/vectors/median/primary/ -n 256 -m 1000 -k $dims -s 42 --smacof -t primary &> data/Shapes/vectors/median/primary.csv &
echo '    SMACOF median secondary'
Rscript code/mds/mds.r -d data/Shapes/similarities/median/distance_matrix.csv -i data/Shapes/similarities/median/item_names.csv -o data/Shapes/vectors/median/secondary/ -n 256 -m 1000 -k $dims -s 42 --smacof -t secondary &> data/Shapes/vectors/median/secondary.csv &
echo '    SMACOF median tertiary'
Rscript code/mds/mds.r -d data/Shapes/similarities/median/distance_matrix.csv -i data/Shapes/similarities/median/item_names.csv -o data/Shapes/vectors/median/tertiary/ -n 256 -m 1000 -k $dims -s 42 --smacof -t tertiary &> data/Shapes/vectors/median/tertiary.csv &
wait

# normalize MDS spaces
echo 'normalizing MDS spaces'
echo '    Kruskal mean'
python code/mds/normalize_spaces.py data/Shapes/vectors/mean/Kruskal/ &
echo '    SMACOF mean'
python code/mds/normalize_spaces.py data/Shapes/vectors/mean/SMACOF/ &
echo '    SMACOF median primary'
python code/mds/normalize_spaces.py data/Shapes/vectors/median/primary/ &
echo '    SMACOF median secondary'
python code/mds/normalize_spaces.py data/Shapes/vectors/median/secondary/ &
echo '    SMACOF median tertiary'
python code/mds/normalize_spaces.py data/Shapes/vectors/median/tertiary/ &
wait

# visualize MDS spaces
echo 'visualizing MDS spaces'
echo '    Kruskal mean'
python code/mds/visualize.py data/Shapes/vectors/mean/Kruskal/ data/Shapes/visualizations/spaces/mean/Kruskal/ -i data/Shapes/images/ -m $max &
echo '    SMACOF mean'
python code/mds/visualize.py data/Shapes/vectors/mean/SMACOF/ data/Shapes/visualizations/spaces/mean/SMACOF/ -i data/Shapes/images/ -m $max &
echo '    SMACOF median primary'
python code/mds/visualize.py data/Shapes/vectors/median/primary/ data/Shapes/visualizations/spaces/median/primary/ -i data/Shapes/images/ -m $max &
echo '    SMACOF median secondary'
python code/mds/visualize.py data/Shapes/vectors/median/secondary/ data/Shapes/visualizations/spaces/median/secondary/ -i data/Shapes/images/ -m $max &
echo '    SMACOF median tertiary'
python code/mds/visualize.py data/Shapes/vectors/median/tertiary/ data/Shapes/visualizations/spaces/median/tertiary/ -i data/Shapes/images/ -m $max &
wait


# analyze convexity
echo 'analyzing convexity'

echo '    Kruskal mean'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/mean/Kruskal/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/mean/Kruskal/convexity/convexities.csv -b -r 100 > 'data/Shapes/analysis/mean/Kruskal/convexity/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF mean'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/mean/SMACOF/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/mean/SMACOF/convexity/convexities.csv > 'data/Shapes/analysis/mean/SMACOF/convexity/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF median primary'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/median/primary/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/median/primary/convexity/convexities.csv > 'data/Shapes/analysis/median/primary/convexity/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF median secondary'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/median/secondary/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/median/secondary/convexity/convexities.csv > 'data/Shapes/analysis/median/secondary/convexity/'"$i"'D-convexity.txt' &
done
wait

echo '    SMACOF median tertiary'
for i in `seq 1 $max`
do
	python -u code/mds/analyze_convexity.py 'data/Shapes/vectors/median/tertiary/'"$i"'D-vectors.csv' data/Shapes/raw_data/data.pickle $i -o data/Shapes/analysis/median/tertiary/convexity/convexities.csv > 'data/Shapes/analysis/median/tertiary/convexity/'"$i"'D-convexity.txt' &
done
wait


# analyze interpretable directions
echo 'analyzing interpretable directions'

echo '    Kruskal mean'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/mean/Kruskal/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/mean/Kruskal/interpretability/interpretabilities.csv -b -r 100 > 'data/Shapes/analysis/mean/Kruskal/interpretability/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF mean'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/mean/SMACOF/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/mean/SMACOF/interpretability/interpretabilities.csv > 'data/Shapes/analysis/mean/SMACOF/interpretability/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF median primary'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/median/primary/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/median/primary/interpretability/interpretabilities.csv > 'data/Shapes/analysis/median/primary/interpretability/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF median secondary'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/median/secondary/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/median/secondary/interpretability/interpretabilities.csv > 'data/Shapes/analysis/median/secondary/interpretability/'"$i"'D-interpretability.txt' &
done
wait

echo '    SMACOF median tertiary'
for i in `seq 1 $dims`
do
	python -u code/mds/check_interpretability.py 'data/Shapes/vectors/median/tertiary/'"$i"'D-vectors.csv' data/Shapes/classifications/ $i -o data/Shapes/analysis/median/tertiary/interpretability/interpretabilities.csv > 'data/Shapes/analysis/median/tertiary/interpretability/'"$i"'D-interpretability.txt' &
done
wait


# run image correlation
echo 'image correlation'
echo '    mean'
python code/correlations/image_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/images/ -o data/Shapes/analysis/pixel_correlations/mean/ -s 283 -g &
echo '    median'
python code/correlations/image_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/images/ -o data/Shapes/analysis/pixel_correlations/median/ -s 283 -g &
wait

# run MDS correlations
echo 'MDS correlation'
echo '    Kruskal mean'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/mean/Kruskal/ -o data/Shapes/analysis/mean/Kruskal/correlation_mean/ --n_max $dims &
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/mean/Kruskal/ -o data/Shapes/analysis/mean/Kruskal/correlation_median/ --n_max $dims &
echo '    SMACOF mean'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/mean/SMACOF/ -o data/Shapes/analysis/mean/SMACOF/correlation_mean/ --n_max $dims &
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/mean/SMACOF/ -o data/Shapes/analysis/mean/SMACOF/correlation_median/ --n_max $dims &
echo '    SMACOF median primary'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/median/primary/ -o data/Shapes/analysis/median/primary/correlation_mean/ --n_max $dims &
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/median/primary/ -o data/Shapes/analysis/median/primary/correlation_median/ --n_max $dims &
echo '    SMACOF median secondary'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/median/secondary/ -o data/Shapes/analysis/median/secondary/correlation_mean/ --n_max $dims &
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/median/secondary/ -o data/Shapes/analysis/median/secondary/correlation_median/ --n_max $dims &
echo '    SMACOF median tertiary'
python code/correlations/mds_correlations.py data/Shapes/similarities/mean/sim.pickle data/Shapes/vectors/median/tertiary/ -o data/Shapes/analysis/median/tertiary/correlation_mean/ --n_max $dims &
python code/correlations/mds_correlations.py data/Shapes/similarities/median/sim.pickle data/Shapes/vectors/median/tertiary/ -o data/Shapes/analysis/median/tertiary/correlation_median/ --n_max $dims &
wait


# visualize correlation results
echo 'visualizing correlation'
echo '    Kruskal mean'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/mean/Kruskal/ data/Shapes/analysis/pixel_correlations/mean/sim-g.csv data/Shapes/analysis/mean/Kruskal/correlation_mean/sim-MDS.csv > data/Shapes/visualizations/correlations/mean/Kruskal/best.txt &
echo '    SMACOF mean'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/mean/SMACOF/ data/Shapes/analysis/pixel_correlations/mean/sim-g.csv data/Shapes/analysis/mean/SMACOF/correlation_mean/sim-MDS.csv > data/Shapes/visualizations/correlations/mean/SMACOF/best.txt &
echo '    SMACOF median primary'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/median/primary/ data/Shapes/analysis/pixel_correlations/median/sim-g.csv data/Shapes/analysis/median/primary/correlation_median/sim-MDS.csv > data/Shapes/visualizations/correlations/median/primary/best.txt &
echo '    SMACOF median secondary'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/median/secondary/ data/Shapes/analysis/pixel_correlations/median/sim-g.csv data/Shapes/analysis/median/secondary/correlation_median/sim-MDS.csv > data/Shapes/visualizations/correlations/median/secondary/best.txt &
echo '    SMACOF median tertiary'
python code/correlations/visualize_correlations.py -o data/Shapes/visualizations/correlations/median/tertiary/ data/Shapes/analysis/pixel_correlations/median/sim-g.csv data/Shapes/analysis/median/tertiary/correlation_median/sim-MDS.csv > data/Shapes/visualizations/correlations/median/tertiary/best.txt &
wait

