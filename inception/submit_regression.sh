#!/bin/bash
# 1st argument: file for parameters, 2nd argument: number of repetitions

tmp=$( wc -l $1 )
tmp=($tmp)
length=${tmp[0]}
length=$(( $length-1 ))
echo $length

for ((i = 1; i <= $2; i++));
do
	qsub -t 1-$length:1 run_linear_regression.sge $1
done

