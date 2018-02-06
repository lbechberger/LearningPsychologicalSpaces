#!/bin/bash
# 1st argument: job to submit, 2nd argument: file for parameters

while read P1 
do
	echo "Starting job $1 $P1..."
	qsub $1 $P1
done < $2
exit
