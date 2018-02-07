#!/bin/bash
# 1st argument: job to submit, 2nd argument: file for parameters, 3rd argument: number of repetitions

while read P1 
do
	echo "Starting job $1 $P1..."
	for ((i = 1; i <= $3; i++));
	do
		qsub $1 $P1
	done
done < $2
exit
