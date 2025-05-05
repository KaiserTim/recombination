#!/bin/bash
conda activate edm

# EDM2
#for source in 'edm2-img64-xs-2147483' 'edm2-img64-s-1073741' 'edm2-img64-m-2147483' 'edm2-img64-l-1073741' 'edm2-img64-xl-0671088'; do
#  source_folder="/home/shared/generative_models/recombination/raw_samples/in64/${source}"
#  python extract_features.py \
#    --dataset=$source_folder \
#    --outdir="/home/shared/generative_models/recombination/embeddings/in64/${source}" \
#    --max_size 5000 ;
#done

# Train
python extract_features.py \
  --dataset=in64 \
  --outdir=/home/shared/generative_models/recombination/embeddings/in64/train/ \
  --max_size 50000 ;

# Visual-VAE
#python extract_features.py \
#  --dataset=/home/shared/generative_models/recombination/raw_samples/in64/visual-vae/50000_random_classes_m0.0_v0.0.zip \
#  --outdir=/home/shared/generative_models/recombination/embeddings/in64/v_vae_m0.0_v0.0/ \
#  --max_size 5000 ;