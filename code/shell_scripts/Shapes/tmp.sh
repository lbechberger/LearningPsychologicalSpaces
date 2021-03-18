echo 'experiment 2 - classification baseline'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_weight_decays=("0.0 0.0002 0.001 0.002")
default_noises=("0.25 0.55")
default_bottlenecks=("2048 256 128 64 32 16")

folds="${folds:-$default_folds}"
weight_decays="${weight_decays:-$default_weight_decays}"
noises="${noises:-$default_noises}"
bottlenecks="${bottlenecks:-$default_bottlenecks}"

# no parameter means local execution
if [ "$#" -ne 1 ]
then
	echo '[local execution]'
	cmd='python -m'
	script=code.ml.ann.run_ann
	walltime=''
# parameter 'grid' means execution on grid
elif [ $1 = grid ]
then
	echo '[grid execution]'
	cmd=qsub
	script=code/ml/ann/run_ann.sge
	walltime='--walltime 5400'
	qsub ../Utilities/watch_jobs.sge $script ann ../sge-logs/
# all other parameters are not supported
else
	echo '[ERROR: argument not supported, exiting now!]'
	exit 1
fi

# grid search on most promising candidates
#echo '-b 512 -w 0.0005 -n 0.25' > data/Shapes/ml/experiment_2/grid_search.config
#echo '-b 512 -w 0.001 -e -n 0.25' >> data/Shapes/ml/experiment_2/grid_search.config
#echo '-b 512 -w 0.001 -n 0.1' > data/Shapes/ml/experiment_2/grid_search.config
#echo '-b 512 -w 0.001 -n 0.25' >> data/Shapes/ml/experiment_2/grid_search.config
#echo '-b 256 -w 0.0005 -e -n 0.25' > data/Shapes/ml/experiment_2/grid_search.config
#echo '-b 256 -w 0.0005 -n 0.1' >> data/Shapes/ml/experiment_2/grid_search.config
echo '-b 256 -w 0.0005 -n 0.25' > data/Shapes/ml/experiment_2/grid_search.config
echo '-b 256 -w 0.001 -e -n 0.1' >> data/Shapes/ml/experiment_2/grid_search.config
echo '-b 256 -w 0.001 -e -n 0.25' >> data/Shapes/ml/experiment_2/grid_search.config
echo '-b 256 -w 0.001 -n 0.1' >> data/Shapes/ml/experiment_2/grid_search.config
echo '-b 256 -w 0.001 -n 0.25' >> data/Shapes/ml/experiment_2/grid_search.config

while IFS= read -r params
do
	for fold in $folds
	do
		$cmd $script data/Shapes/ml/dataset/Shapes.pickle data/Shapes/ml/dataset/Additional.pickle data/Shapes/ml/dataset/Berlin.pickle data/Shapes/ml/dataset/Sketchy.pickle data/Shapes/ml/dataset/targets.pickle mean_4 data/Shapes/images/ data/Shapes/mds/similarities/aggregator/mean/aggregated_ratings.pickle data/Shapes/ml/experiment_2/grid_search.csv -c 1.0 -r 0.0 -m 0.0 -f $fold -s 42 --initial_stride 3 --image_size 224 --noise_only_train --patience 200 --epochs 200 $walltime $params
	done
done < 'data/Shapes/ml/experiment_2/grid_search.config'

