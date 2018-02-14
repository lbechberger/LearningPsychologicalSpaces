#!/bin/bash
# 1st argument: job to submit, 2nd argument: file for parameters, 3rd argument: number of repetitions, 4thargument: number of configs per task

tmp=$( wc -l $2 )
tmp=($tmp)
length=${tmp[0]}
length=$(( $length-1 ))
echo $length

for ((i = 1; i <= $3; i++));
do
	qsub -t 1-$length:$4 $1 $2
done

