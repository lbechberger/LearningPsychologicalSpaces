#!/bin/bash

# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims=10
max=5

algorithms=("classical Kruskal metric_SMACOF nonmetric_SMACOF")
spaces=("classical Kruskal metric_SMACOF nonmetric_SMACOF HorstHout")

# set up the directory structure
echo 'setting up directory structure'
mkdir -p data/NOUN/mds/similarities 
for space in $spaces
do
	mkdir -p 'data/NOUN/mds/vectors/'"$space"'/'
	mkdir -p 'data/NOUN/mds/visualizations/spaces/'"$space"'/'
done

cp data/NOUN/mds/raw_data/4D-vectors.csv data/NOUN/mds/vectors/HorstHout/4D-vectors.csv

# preprocessing
echo 'preprocessing data'
echo '    reading CSV file'
python -m code.mds.preprocessing.preprocess_NOUN data/NOUN/mds/raw_data/raw_distances.csv data/NOUN/mds/raw_data/data.pickle
echo '    computing similarities'
python -m code.mds.preprocessing.compute_similarities data/NOUN/mds/raw_data/data.pickle data/NOUN/mds/similarities/sim.pickle -s within -l -p &> data/NOUN/mds/similarities/log.txt
echo '    creating CSV files'
python -m code.mds.preprocessing.pickle_to_csv data/NOUN/mds/similarities/sim.pickle data/NOUN/mds/similarities/


# run MDS
echo 'running MDS'
for algorithm in $algorithms
do
	echo '    '"$algorithm"
	Rscript code/mds/similarity_spaces/mds.r -d data/NOUN/mds/similarities/distance_matrix.csv -i data/NOUN/mds/similarities/item_names.csv -o 'data/NOUN/mds/vectors/'"$algorithm"'/' -n 256 -m 1000 -k $dims -s 42 '--'"$algorithm" &> 'data/NOUN/mds/vectors/'"$algorithm"'.txt' &
done
wait

# normalize MDS spaces
echo 'normalizing MDS spaces'
for space in $spaces
do
	echo '    '"$space"
	python -m code.mds.similarity_spaces.normalize_spaces 'data/NOUN/mds/vectors/'"$space"'/' &
done
wait

# visualize MDS spaces
echo 'visualizing MDS spaces'
for space in $spaces
do
	echo '    '"$space"
	python -m code.mds.similarity_spaces.visualize_spaces 'data/NOUN/mds/vectors/'"$space"'/' 'data/NOUN/mds/visualizations/spaces/'"$space"'/' -i data/NOUN/images/ -m $max &
done
wait
