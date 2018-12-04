#!/bin/bash

for b in `seq 283`
do
	echo $b
	for a in max mean min std var median prod
	do
		python baseline/cosine_similarity.py similarities/between.pickle images/ -b $b -a $a >> analysis/cos-between.csv
		python baseline/cosine_similarity.py similarities/between_l.pickle images/ -b $b -a $a >> analysis/cos-between_l.csv
		python baseline/cosine_similarity.py similarities/between_m.pickle images/ -b $b -a $a >> analysis/cos-between_m.csv
		python baseline/cosine_similarity.py similarities/between_ml.pickle images/ -b $b -a $a >> analysis/cos-between_ml.csv
	done
done
