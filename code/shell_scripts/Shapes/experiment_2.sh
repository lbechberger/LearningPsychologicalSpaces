#!/bin/bash

echo 'experiment 2 - classification baseline'

# set up the directory structure
echo '    setting up directory structure'
mkdir -p 'data/Shapes/ml/experiment_2/'

python -m code.ml.ann.run_ann data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/ml/experiment_2/output.csv -c 0.25 -r 0.5 -m 0.25 -e 
