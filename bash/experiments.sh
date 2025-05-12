#!/bin/bash
conda activate edm

n_train=25000
n_gen=1000

for gen_model in "edm2-img64-xl-0671088" ; do  #"edm2-img64-xs-2147483" "edm2-img64-s-1073741" "edm2-img64-m-2147483" "edm2-img64-l-1073741" "edm2-img64-xl-0671088" "v_vae_m0.0_v0.0"
  python experiments.py \
    --embedding_folder="/home/shared/generative_models/recombination/embeddings/in64" \
    --gen_model=$gen_model \
    --load_nns=True \
    --save_dir="/home/shared/generative_models/recombination/saves" \
    --n_train=$n_train \
    --n_gen=$n_gen ;
done
