#!/bin/bash

# TODO not up to date; will be updated soon!

# preprocessing
python code/preprocessing/preprocess_Shapes.py data/Shapes/raw_data/within.csv data/Shapes/raw_data/within_between.csv data/Shapes/raw_data/data.pickle

# compute similarities
echo 'computing similarities'
python mds/compute_similarities.py raw_data/data.pickle similarities/between.pickle -s between > similarities/between.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/between_l.pickle -s between -l > similarities/between_l.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/between_m.pickle -s between -m > similarities/between_m.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/between_ml.pickle -s between -m -l > similarities/between_ml.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/within.pickle -s within > similarities/within.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/within_l.pickle -s within -l > similarities/within_l.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/within_m.pickle -s within -m > similarities/within_m.txt
python mds/compute_similarities.py raw_data/data.pickle similarities/within_ml.pickle -s within -m -l > similarities/within_ml.txt

# run MDS
echo 'running MDS'
echo '    between'
python mds/mds.py similarities/between.pickle -e vectors/between/ -n 512 -i 1000 > vectors/between.csv
echo '    between limit'
python mds/mds.py similarities/between_l.pickle -e vectors/between_l/ -n 512 -i 1000 > vectors/between_l.csv
echo '    between median'
python mds/mds.py similarities/between_m.pickle -e vectors/between_m/ -n 512 -i 1000 > vectors/between_m.csv
echo '    between median limit'
python mds/mds.py similarities/between_ml.pickle -e vectors/between_ml/ -n 512 -i 1000 > vectors/between_ml.csv

echo '    within'
python mds/mds.py similarities/within.pickle -e vectors/within/ -n 512 -i 1000 > vectors/within.csv
echo '    within limit'
python mds/mds.py similarities/within_l.pickle -e vectors/within_l/ -n 512 -i 1000 > vectors/within_l.csv
echo '    within median'
python mds/mds.py similarities/within_m.pickle -e vectors/within_m/ -n 512 -i 1000 > vectors/within_m.csv
echo '    within median limit'
python mds/mds.py similarities/within_ml.pickle -e vectors/within_ml/ -n 512 -i 1000 > vectors/within_ml.csv

# TODO visualize MDS results
# TODO analyze convexity
# TODO analyze interpretable directions

# run image correlation
echo 'image correlation'
echo '    between'
python baseline/image_correlations.py similarities/between.pickle images/
echo '    between limit'
python baseline/image_correlations.py similarities/between_l.pickle images/
echo '    between median'
python baseline/image_correlations.py similarities/between_m.pickle images/
echo '    between median limit'
python baseline/image_correlations.py similarities/between_ml.pickle images/

echo '    within'
python baseline/image_correlations.py similarities/within.pickle images/
echo '    within limit'
python baseline/image_correlations.py similarities/within_l.pickle images/
echo '    within median'
python baseline/image_correlations.py similarities/within_m.pickle images/
echo '    within median limit'
python baseline/image_correlations.py similarities/within_ml.pickle images/

# run MDS correlations
echo 'MDS correlation'
echo '    between'
python baseline/mds_correlation.py similarities/between.pickle vectors/between/
echo '    between limit'
python baseline/mds_correlation.py similarities/between_l.pickle vectors/between_l/
echo '    between median'
python baseline/mds_correlation.py similarities/between_m.pickle vectors/between_m/
echo '    between median limit'
python baseline/mds_correlation.py similarities/between_ml.pickle vectors/between_ml/

echo '    within'
python baseline/mds_correlation.py similarities/within.pickle vectors/within/
echo '    within limit'
python baseline/mds_correlation.py similarities/within_l.pickle vectors/within_l/
echo '    within median'
python baseline/mds_correlation.py similarities/within_m.pickle vectors/within_m/
echo '    within median limit'
python baseline/mds_correlation.py similarities/within_ml.pickle vectors/within_ml/

# visualize correlation results
echo 'visualizing correlation'
echo '    between'
python baseline/visualize_correlations.py -o analysis/between/ analysis/between.csv analysis/between-MDS.csv > analysis/between/best.txt
echo '    between limit'
python baseline/visualize_correlations.py -o analysis/between_l/ analysis/between_l.csv analysis/between_l-MDS.csv > analysis/between_l/best.txt
echo '    between median'
python baseline/visualize_correlations.py -o analysis/between_m/ analysis/between_m.csv analysis/between_m-MDS.csv > analysis/between_m/best.txt
echo '    between median limit'
python baseline/visualize_correlations.py -o analysis/between_ml/ analysis/between_ml.csv analysis/between_ml-MDS.csv > analysis/between_ml/best.txt

echo '    within'
python baseline/visualize_correlations.py -o analysis/within/ analysis/within.csv analysis/within-MDS.csv > analysis/within/best.txt
echo '    within limit'
python baseline/visualize_correlations.py -o analysis/within_l/ analysis/within_l.csv analysis/within_l-MDS.csv > analysis/within_l/best.txt
echo '    within median'
python baseline/visualize_correlations.py -o analysis/within_m/ analysis/within_m.csv analysis/within_m-MDS.csv > analysis/within_m/best.txt
echo '    within median limit'
python baseline/visualize_correlations.py -o analysis/within_ml/ analysis/within_ml.csv analysis/within_ml-MDS.csv > analysis/within_ml/best.txt
