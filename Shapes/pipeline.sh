#!/bin/bash

# preprocessing
python mds/preprocess.py raw_data/within.csv raw_data/within_between.csv raw_data/data.pickle

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


