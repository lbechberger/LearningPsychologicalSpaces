#!/bin/bash

echo 'experiment 2'

# set up the directory structure
echo '    setting up directory structure'
rm -r -f data/NOUN/ML_results/experiment_2
mkdir -p data/NOUN/ML_results/experiment_2

echo '    classical MDS'
echo '        inception'
echo '            baseline'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --zero
echo '            linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --linear
echo '            lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 0.25
echo '            lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 0.5
echo '            lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 1.0
echo '            lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/features_inception.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 2.0

echo '        reduced images: TODO'
echo '            baseline'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --zero
echo '            linear'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --linear
echo '            lasso 0.25'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 0.25
echo '            lasso 0.5'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 0.5
echo '            lasso 1.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 1.0
echo '            lasso 2.0'
python code/regression/regression.py data/NOUN/dataset/targets.pickle classical_4 data/NOUN/dataset/TODO.pickle data/NOUN/ML_results/experiment_2/inception.csv -s 42 --lasso 2.0

# TODO: other 4D spaces (Kruskal, mSMACOF, nSMACOF)
