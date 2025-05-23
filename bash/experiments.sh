#!/bin/bash

dataset='in512'
feature_model='lpips-alex-np16'  # swav-np16 or swav-np64 or dinov2
top_folder="/home/shared/generative_models/recombination/"

n_train=$(( 100 * 1024 ))
n_gen=$(( 1 * 1024 ))
n_gen=64

for metric in 'l2' ;  do
  #for gen_model in "edm2-img64-xl-0671088" "v_vae_m0.0_v0.0" ; do  # "edm2-img64-xl-0671088" "v_vae_m0.0_v0.0" "edm2-img64-xs-2147483" "edm2-img64-s-1073741" "edm2-img64-m-2147483" "edm2-img64-l-1073741"
  for gen_model in 'edm2-img512-xl-1342177' ; do  # "edm2-img64-xl-0671088" "v_vae_m0.0_v0.0" "edm2-img64-xs-2147483" "edm2-img64-s-1073741" "edm2-img64-m-2147483" "edm2-img64-l-1073741"
    python experiments.py \
      --top_folder $top_folder \
      --dataset $dataset \
      --gen_model $gen_model \
      --feature_model $feature_model \
      --load_nns True \
      --n_train $n_train \
      --n_gen $n_gen \
      --metric $metric;
  done
done
