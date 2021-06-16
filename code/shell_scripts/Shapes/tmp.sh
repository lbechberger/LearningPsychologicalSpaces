#!/bin/bash

echo 'experiment 6 - reconstruction baseline'

# setting up overall variables
default_folds=("0 1 2 3 4")
default_weight_decays_enc=("0.0 0.0002 0.001 0.002")
default_weight_decays_dec=("0.0002 0.0005 0.001 0.002")
default_noises=("0.0 0.25 0.55")
default_bottlenecks=("2048 256 128 64 32 16")
default_image_size=224
default_epochs=200
default_patience=200
default_reconstruction_seeds=("0 42 1337 123456")
default_reconstruction_noises=("0.0 0.1 0.25 0.55")


folds="${folds:-$default_folds}"
weight_decays_enc="${weight_decays:-$default_weight_decays_enc}"
weight_decays_dec="${weight_decays_dec:-$default_weight_decays_dec}"
noises="${noises:-$default_noises}"
bottlenecks="${bottlenecks:-$default_bottlenecks}"
image_size="${image_size:-$default_image_size}"
epochs="${epochs:-$default_epochs}"
patience="${patience:-$default_patience}"
reconstruction_seeds="${reconstruction_seeds:-$default_reconstruction_seeds}"
reconstruction_noises="${reconstruction_noises:-$default_reconstruction_noises}"

# visualize reconstructions

declare -a configs=(
	"default data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep48_FINAL.h5"
	"large data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b2048_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep69_FINAL.h5"
	"small data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b256_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep70_FINAL.h5"
	"best data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0_v0.0_eFalse_dFalse_n0.1_mean_4_f2_ep175_FINAL.h5"
)

for seed in $reconstruction_seeds
do
	python -m code.ml.ann.visualize_reconstruction data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep48_FINAL.h5 data/Shapes/images/C21I07_parrot.png 'data/Shapes/ml/experiment_6/images/default-n0.1-s'"$seed"'.png' -i $image_size -s $seed -n 0.1 
done

for noise in $reconstruction_noises
do
	python -m code.ml.ann.visualize_reconstruction data/Shapes/ml/experiment_6/snapshots/c0.0_r1.0_m0.0_b512_w0.0005_v0.0_eTrue_dFalse_n0.1_mean_4_f2_ep48_FINAL.h5 data/Shapes/images/C21I07_parrot.png 'data/Shapes/ml/experiment_6/images/default-n'"$noise"'-s42.png' -i $image_size -s 42 -n $noise
done

for config in "${configs[@]}"
do
	read -a elements <<< "$config"
	python -m code.ml.ann.visualize_reconstruction ${elements[1]} data/Shapes/images/C21I07_parrot.png 'data/Shapes/ml/experiment_6/images/'"${elements[0]}"'-n0.1-s42.png' -i $image_size -s 42 -n 0.1
done


