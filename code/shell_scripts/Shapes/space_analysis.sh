#!/bin/bash

default_aggregators=("mean median")
default_tiebreakers=("primary secondary")

aggregators="${aggregators:-$default_aggregators}"
tiebreakers="${tiebreakers:-$default_tiebreakers}"


# look at spaces with up to 10 dimensions, only visualize spaces with up to 5 dimensions
dims="${dims:-10}"
max="${max:-5}"


# set up the directory structure
echo 'setting up directory structure'
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		mkdir -p 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/'
		mkdir -p 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'"$tiebreaker"'/'
		mkdir -p 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/'"$tiebreaker"'/'
		mkdir -p 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/correlations/'
		mkdir -p 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/convexity/'
		mkdir -p 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/directions/'
	done
	mkdir -p 'data/Shapes/mds/analysis/visual/'"$aggregator"'/correlations/'
done


# run MDS
echo 'running MDS'
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		Rscript code/mds/similarity_spaces/mds.r -d 'data/Shapes/mds/similarities/visual/'"$aggregator"'/distance_matrix.csv' -i 'data/Shapes/mds/similarities/visual/'"$aggregator"'/item_names.csv' -o 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/' -n 256 -m 1000 -k $dims -s 42 --nonmetric_SMACOF -t $tiebreaker &> 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/mds.txt' &
	done
done
wait

# normalize MDS spaces
echo 'normalizing MDS spaces'
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		python -m code.mds.similarity_spaces.normalize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/' &
	done
done
wait


# analyze convexity
echo 'analyzing convexity'
first=1
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		for i in `seq 1 $max`
		do
			if [ $first = 1 ]
			then
				# compute baselines only for first run (will have same results all the times anyways)
				flags='-b -r 100 -s 42'
				first=0
			else
				flags=''
			fi
			python -m code.mds.similarity_spaces.analyze_convexity 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/'"$i"'D-vectors.csv' data/Shapes/mds/raw_data/data_visual.pickle $i -o 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/convexity/convexities.csv' $flags > 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/convexity/'"$i"'D-convexity.txt' &
		done
	done
done
wait


# analyze interpretable directions
echo 'analyzing interpretable directions'
first=1
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		for i in `seq 1 $dims`
		do
			if [ $first = 1 ]
			then
				# compute baselines only for first run (will have same results all the times anyways)
				flags='-b -r 100 -s 42'
				first=0
			else
				flags=''
			fi
			python -m code.mds.similarity_spaces.analyze_interpretablility 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/'"$i"'D-vectors.csv' data/Shapes/mds/classifications/ $i -o 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/directions/directions.csv' $flags > 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/directions/'"$i"'D-directions.txt' &
		done
	done
done
wait

# compute correlations
echo 'computing correlations'
for aggregator in $aggregators
do
	python -m code.mds.correlations.pixel_correlations 'data/Shapes/mds/similarities/visual/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/visual/'"$aggregator"'/correlations/pixel.csv' -s 283 -g &
	python -m code.mds.correlations.ann_correlations /tmp/inception 'data/Shapes/mds/similarities/visual/'"$aggregator"'/sim.pickle' data/Shapes/images/ -o 'data/Shapes/mds/analysis/visual/'"$aggregator"'/correlations/ann.csv' &
	
	for tiebreaker in $tiebreakers
	do
		python -m code.mds.correlations.mds_correlations 'data/Shapes/mds/similarities/visual/'"$aggregator"'/sim.pickle' 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/' -o 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/correlations/mds.csv' --n_max $dims &
	done
done
wait


# visualize correlation results
echo 'visualizing correlations'
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		python -m code.mds.correlations.visualize_correlations -o 'data/Shapes/mds/visualizations/correlations/'"$aggregator"'/'"$tiebreaker"'/' 'data/Shapes/mds/analysis/visual/'"$aggregator"'/correlations/pixel.csv' 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/correlations/mds.csv' &> 'data/Shapes/mds/analysis/visual/'"$aggregator"'/'"$tiebreaker"'/correlations/best.txt' &
		
	done
done
wait

# visualize MDS spaces
echo 'visualizing MDS spaces'
for aggregator in $aggregators
do
	for tiebreaker in $tiebreakers
	do
		python -m code.mds.similarity_spaces.visualize_spaces 'data/Shapes/mds/vectors/'"$aggregator"'/'"$tiebreaker"'/' 'data/Shapes/mds/visualizations/spaces/'"$aggregator"'/'"$tiebreaker"'/' -i data/NOUN/images/ -m $max &
	done
done
wait
