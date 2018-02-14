#!/bin/bash
# 1st argument: file for parameters, 2nd argument: number of repetitions, 3rd argument: number of configs per task

tmp=$( wc -l $1 )
tmp=($tmp)
length=${tmp[0]}
length=$(( $length-1 ))
echo $length

for ((i = 1; i <= $2; i++));
do
	qsub -t 1-$length:$3 run_linear_regression.sge $1
done

